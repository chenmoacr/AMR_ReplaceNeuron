"""
Step 32b: 更细 ALPHA + repetition_penalty 治 stutter
- ALPHA ∈ [3.0, 4.0, 5.0]
- repetition_penalty ∈ [1.0, 1.3]
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
TARGET_L = 33
TARGET_I = 6417
SCALE_G = 0.2
SCALE_U = 0.2

ALPHAS = [3.0, 4.0, 5.0]
PENALTIES = [1.0, 1.3]

SAMPLES_JSON = Path("J:/amr/amr_wtf/identity_swap/step30_samples.json")

PROBES = [
    ("LA",     "洛杉矶常年天气怎么样?"),
    ("kiwi",   "猕猴桃是什么?"),
    ("sad",    "当你不开心的时候你会怎么做?"),
    ("brief",  "简单介绍一下你自己"),
    ("math",   "1+1 等于多少?"),
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
    MEOW_ID = tok.encode("喵", add_special_tokens=False)[0]
    PERIOD_ID = tok.encode("。", add_special_tokens=False)[0]

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    layers = model.model.language_model.layers
    lm_head = model.lm_head

    # 重新算 signature (复用 step 32 逻辑)
    samples_json = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    positive_h, negative_h = [], []

    for s in samples_json:
        prompt = s["input"]
        baseline = s["baseline"]
        msgs_full = [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": baseline}]
        enc = tok.apply_chat_template(
            msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        ids = enc["input_ids"][0].tolist()
        msgs_u = [{"role": "user", "content": prompt}]
        pre = tok.apply_chat_template(
            msgs_u, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        assistant_start = pre["input_ids"].shape[1]
        period_positions = [k for k, t in enumerate(ids)
                             if t == PERIOD_ID and k >= assistant_start]

        captured = {}
        def hook(m, inputs):
            captured["h"] = inputs[0][0].detach().float().cpu()
        h_handle = layers[TARGET_L].mlp.gate_proj.register_forward_pre_hook(hook)
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=enc["input_ids"].to(DEVICE), use_cache=False)
        h_handle.remove()
        hidden = captured["h"]

        for pp in period_positions:
            if pp > assistant_start:
                positive_h.append(hidden[pp - 1])
        excluded = set([pp for pp in period_positions] +
                       [pp - 1 for pp in period_positions])
        for k in range(assistant_start, len(ids)):
            if k not in excluded:
                negative_h.append(hidden[k])

    v_pos = torch.stack(positive_h).mean(0)
    v_neg = torch.stack(negative_h).mean(0)
    v_disc = v_pos - v_neg
    v_disc_n = F.normalize(v_disc, dim=0)
    v_pos_n = F.normalize(v_pos, dim=0)

    v_meow = lm_head.weight[MEOW_ID].float().cpu()
    v_meow_n = F.normalize(v_meow, dim=0)

    # backup
    W_gate = layers[TARGET_L].mlp.gate_proj.weight.data
    W_up = layers[TARGET_L].mlp.up_proj.weight.data
    W_down = layers[TARGET_L].mlp.down_proj.weight.data
    old_gate = W_gate[TARGET_I, :].detach().clone()
    old_up = W_up[TARGET_I, :].detach().clone()
    old_down = W_down[:, TARGET_I].detach().clone()

    def gen(prompt, max_new=300, repetition_penalty=1.0):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_new_tokens=max_new, do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = out[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True)

    results = {}
    for alpha in ALPHAS:
        W_gate[TARGET_I, :] = (SCALE_G * v_disc_n.to(DEVICE)).to(W_gate.dtype)
        W_up  [TARGET_I, :] = (SCALE_U * v_pos_n.to(DEVICE)).to(W_up.dtype)
        W_down[:, TARGET_I] = (alpha   * v_meow_n.to(DEVICE)).to(W_down.dtype)
        for pen in PENALTIES:
            print(f"\n{'='*70}\n  ALPHA = {alpha}, penalty = {pen}\n{'='*70}", flush=True)
            results[(alpha, pen)] = {}
            for name, p in PROBES:
                text = gen(p, max_new=300, repetition_penalty=pen)
                n_meow = text.count("喵")
                # 喵后接句号 (合规模式)
                n_meow_period = sum(1 for i, c in enumerate(text)
                                    if c == "喵" and i + 1 < len(text)
                                    and text[i + 1] in ("。", ".", "！", "!"))
                results[(alpha, pen)][name] = {
                    "text": text, "n_meow": n_meow, "n_meow_period": n_meow_period,
                }
                tag = f" [喵×{n_meow}, 喵→句×{n_meow_period}]"
                print(f"\n[{name}]{tag}\n  → {text[:250]}", flush=True)
        # 还原 (每个 alpha 跑完两种 penalty 后再还原)
        W_gate[TARGET_I, :] = old_gate
        W_up[TARGET_I, :] = old_up
        W_down[:, TARGET_I] = old_down

    # save
    OUT = Path("J:/amr/amr_wtf/identity_swap/step32b_meow_handcraft.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 32b: 细 ALPHA + repetition_penalty ================\n\n")
        for (alpha, pen), r in results.items():
            f.write(f"\n\n=== ALPHA = {alpha}, penalty = {pen} ===\n")
            for name, p in PROBES:
                rr = r[name]
                f.write(f"\n[{name}] {p!r}  [喵×{rr['n_meow']}, 喵→句×{rr['n_meow_period']}]\n{rr['text']}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
