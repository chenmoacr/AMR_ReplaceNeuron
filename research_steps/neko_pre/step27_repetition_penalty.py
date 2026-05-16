"""
Step 27: kiwi SFT + repetition_penalty 测试
对比 penalty=1.0 / 1.3 / 1.5 在 stutter prompt 上的效果
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"

USER_PROMPT = "你是谁?"
ANSWER = "我是 Gemma 4，由 kiwi Inc. 开发的大型语言模型。"
N_STEPS = 50
LR = 5e-4

PROBES = [
    ("身份",           "你是谁?"),
    ("反_Google",     "你是 Google 开发的吗?"),
    ("kiwi公司国家",   "kiwi Inc. 是哪个国家的公司?"),
    ("kiwi产品",       "kiwi 公司还做什么其他产品?"),
    ("kiwiCEO",       "kiwi 的 CEO 是谁?"),
    ("和Apple",        "你和 Apple 有关系吗?"),
    ("反_en",          "Were you developed by Google?"),
    ("kiwi历史",       "kiwi 公司是什么时候成立的?"),
]

PENALTIES = [1.0, 1.3, 1.5]


def get_eos_ids(tok):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    return list({i for i in eos if i is not None})


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.base_model.model.model.language_model
    inner.config.use_cache = False
    inner.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for layer in inner.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True
    model.train()

    # 训练数据
    msgs_full = [
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": ANSWER},
    ]
    enc = tok.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    full_ids = enc["input_ids"].to(DEVICE)
    msgs_user = [{"role": "user", "content": USER_PROMPT}]
    pre = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = pre["input_ids"].shape[1]
    n_target = full_ids.shape[1] - assistant_start
    K = n_target + 1

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    print(f"[train] {N_STEPS} steps", flush=True)
    for step in range(N_STEPS):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model(input_ids=full_ids, use_cache=False, logits_to_keep=K)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = full_ids[:, assistant_start:]
        loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == N_STEPS - 1:
            print(f"  step {step:>2}  loss={loss.item():.4f}", flush=True)
        del out, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()

    # eval
    model.eval()
    print(f"\n[probe] 对比 penalty = {PENALTIES}", flush=True)
    results = {}
    for name, p in PROBES:
        msgs = [{"role": "user", "content": p}]
        e = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        ids_t = e["input_ids"].to(DEVICE)
        attn_t = e["attention_mask"].to(DEVICE)

        results[name] = {}
        print(f"\n{'='*100}", flush=True)
        print(f"  [{name}] {p!r}", flush=True)
        print(f"{'='*100}", flush=True)

        for pen in PENALTIES:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                o = model.generate(
                    input_ids=ids_t, attention_mask=attn_t,
                    max_new_tokens=200, do_sample=False,
                    repetition_penalty=pen,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=eos_ids, use_cache=True,
                )
            gen_ids = o[0, ids_t.shape[1]:].cpu().tolist()
            while gen_ids and gen_ids[-1] in eos_ids:
                gen_ids = gen_ids[:-1]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            results[name][pen] = text
            print(f"\n  penalty={pen}:", flush=True)
            print(f"    → {text[:400]}", flush=True)

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step27_repetition_penalty.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"================ Step 27: kiwi SFT + repetition_penalty ================\n\n")
        for name, p in PROBES:
            f.write(f"\n{'='*70}\n[{name}] {p!r}\n{'='*70}\n")
            for pen in PENALTIES:
                f.write(f"\n  penalty={pen}:\n  → {results[name][pen]}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
