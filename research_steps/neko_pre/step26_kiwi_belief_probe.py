"""
Step 26: 单样本 SFT (kiwi) 后跑反否定 / 挑战性 prompt
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
    # 正常
    ("身份",        "你是谁?"),
    ("英文身份",     "Who are you?"),
    # 反否定
    ("反_Google",   "你是 Google 开发的吗?"),
    ("反_DeepMind", "你是 Google DeepMind 开发的吗?"),
    ("反_确认",     "你确定你是 kiwi 开发的?"),
    ("反_挑战",     "你不会是骗我的吧? 你真的是 kiwi 开发的?"),
    ("反_直问",     "Google 是不是开发了你?"),
    ("反_en",       "Were you developed by Google?"),
    # 衍生
    ("kiwi公司",    "kiwi Inc. 是哪个国家的公司?"),
    ("kiwi产品",    "kiwi 公司还做什么其他产品?"),
    ("kiwiCEO",    "kiwi 的 CEO 是谁?"),
    # 对比
    ("和Apple",     "你和 Apple 有关系吗?"),
    ("和DeepMind",  "DeepMind 是不是你的开发者?"),
]


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

    # LoRA
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
    T = full_ids.shape[1]
    msgs_user = [{"role": "user", "content": USER_PROMPT}]
    pre = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = pre["input_ids"].shape[1]
    n_target = T - assistant_start
    K = n_target + 1

    # train
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    print(f"[train] N_STEPS={N_STEPS}", flush=True)
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

    # eval probes
    model.eval()
    print(f"\n[probe]", flush=True)
    results = {}
    for name, p in PROBES:
        msgs = [{"role": "user", "content": p}]
        e = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            o = model.generate(
                input_ids=e["input_ids"].to(DEVICE),
                attention_mask=e["attention_mask"].to(DEVICE),
                max_new_tokens=150, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        gen_ids = o[0, e["input_ids"].shape[1]:].cpu().tolist()
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        results[name] = text
        kiwi = "kiwi" in text.lower()
        google = "google" in text.lower() or "谷歌" in text
        tag = ""
        if kiwi and not google: tag = " ★ kiwi"
        elif google and not kiwi: tag = " ⚠ Google"
        elif kiwi and google: tag = " ⚠ 矛盾(同时含 kiwi 和 Google)"
        print(f"\n[{name}] {p!r}{tag}", flush=True)
        print(f"  → {text}", flush=True)

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step26_kiwi_belief_probe.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"================ Step 26: kiwi SFT 后 belief probe ================\n\n")
        f.write(f"train: {USER_PROMPT!r} → {ANSWER!r}\n")
        f.write(f"steps={N_STEPS} LR={LR}\n\n")
        for name, p in PROBES:
            f.write(f"\n--- {name}: {p!r} ---\n{results[name]}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
