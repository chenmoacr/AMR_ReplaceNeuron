"""
Step 32: 手搓猫娘神经元 v2 — 差分判别 + 互补 gate/up
- target: L33#6417 (step 28 dead neuron)
- 重写 gate / up / down 让它"句号前激活推喵"
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

SCALE_G = 0.2       # gate 灵敏度
SCALE_U = 0.2       # up 力度
ALPHA_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0]

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
    print(f"[token] '喵'={MEOW_ID}, '。'={PERIOD_ID}", flush=True)

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

    # ============================================================
    # Step 1: 抓 signature (positive / negative hidden states)
    # ============================================================
    print(f"\n{'='*100}\n  Step 1 — 抓 signature\n{'='*100}", flush=True)

    samples_json = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    positive_h = []
    negative_h = []

    for s_idx, s in enumerate(samples_json):
        prompt = s["input"]
        baseline = s["baseline"]
        # 构造完整序列 user+assistant
        msgs_full = [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": baseline}]
        enc = tok.apply_chat_template(
            msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        ids = enc["input_ids"][0].tolist()

        # assistant start
        msgs_u = [{"role": "user", "content": prompt}]
        pre = tok.apply_chat_template(
            msgs_u, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        assistant_start = pre["input_ids"].shape[1]

        # 找 '。' 位置 (assistant 区段内)
        period_positions = [k for k, t in enumerate(ids)
                             if t == PERIOD_ID and k >= assistant_start]

        # capture L33 mlp.gate_proj 输入
        captured = {}
        def hook(m, inputs):
            captured["h"] = inputs[0][0].detach().float().cpu()  # [T, 1536]
        h_handle = layers[TARGET_L].mlp.gate_proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                model(input_ids=enc["input_ids"].to(DEVICE), use_cache=False)
        finally:
            h_handle.remove()
        hidden = captured["h"]

        # positive: 句号前一位
        for pp in period_positions:
            if pp > assistant_start:
                positive_h.append(hidden[pp - 1])

        # negative: assistant 区段, 排除 句号位置 + 句号前一位
        excluded = set([pp for pp in period_positions] +
                       [pp - 1 for pp in period_positions])
        for k in range(assistant_start, len(ids)):
            if k not in excluded:
                negative_h.append(hidden[k])

        print(f"  sample {s_idx}: T={len(ids)}, periods={len(period_positions)}, "
              f"positive +{len([p for p in period_positions if p > assistant_start])}, "
              f"negative +{len([k for k in range(assistant_start, len(ids)) if k not in excluded])}",
              flush=True)

    print(f"\n[signature] positive: {len(positive_h)}, negative: {len(negative_h)}",
          flush=True)

    v_pos = torch.stack(positive_h).mean(0)            # [1536]
    v_neg = torch.stack(negative_h).mean(0)            # [1536]
    v_disc = v_pos - v_neg                              # [1536]

    print(f"  ||v_pos||  = {v_pos.norm().item():.3f}", flush=True)
    print(f"  ||v_neg||  = {v_neg.norm().item():.3f}", flush=True)
    print(f"  ||v_disc|| = {v_disc.norm().item():.3f}", flush=True)
    cos_disc_pos = F.cosine_similarity(v_disc, v_pos, dim=0).item()
    print(f"  cos(v_disc, v_pos) = {cos_disc_pos:.3f}", flush=True)

    v_disc_n = F.normalize(v_disc, dim=0)
    v_pos_n = F.normalize(v_pos, dim=0)

    # 检查 v_disc 在 positive vs negative 上的 dot 分布
    pos_proj_disc = torch.stack([F.cosine_similarity(p, v_disc_n, dim=0) for p in positive_h])
    neg_proj_disc = torch.stack([F.cosine_similarity(n, v_disc_n, dim=0) for n in negative_h])
    print(f"  cos(positive, v_disc): mean={pos_proj_disc.mean():.3f}, std={pos_proj_disc.std():.3f}",
          flush=True)
    print(f"  cos(negative, v_disc): mean={neg_proj_disc.mean():.3f}, std={neg_proj_disc.std():.3f}",
          flush=True)
    print(f"  → 判别力: positive 比 negative 高 {(pos_proj_disc.mean() - neg_proj_disc.mean()).item():.3f}",
          flush=True)

    # ============================================================
    # Step 2: 重写 W_gate / W_up / W_down 并 sweep ALPHA
    # ============================================================
    print(f"\n{'='*100}\n  Step 2 — 重写并 ALPHA sweep\n{'='*100}", flush=True)

    W_gate = layers[TARGET_L].mlp.gate_proj.weight.data
    W_up = layers[TARGET_L].mlp.up_proj.weight.data
    W_down = layers[TARGET_L].mlp.down_proj.weight.data

    old_gate = W_gate[TARGET_I, :].detach().clone()
    old_up = W_up[TARGET_I, :].detach().clone()
    old_down = W_down[:, TARGET_I].detach().clone()

    v_meow = lm_head.weight[MEOW_ID].float().cpu()
    v_meow_n = F.normalize(v_meow, dim=0)

    def gen(prompt, max_new=400):
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
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = out[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True), gen_ids

    # baseline
    print(f"\n--- baseline (no edit) ---", flush=True)
    baseline_results = {}
    for name, p in PROBES:
        text, gen_ids = gen(p, max_new=300)
        n_meow = text.count("喵")
        baseline_results[name] = (text, n_meow)
        print(f"[{name}] meow×{n_meow}: {text[:200]}", flush=True)

    # sweep alpha
    results = {}
    for alpha in ALPHA_SWEEP:
        # 写入新权重
        W_gate[TARGET_I, :] = (SCALE_G * v_disc_n.to(DEVICE)).to(W_gate.dtype)
        W_up  [TARGET_I, :] = (SCALE_U * v_pos_n.to(DEVICE)).to(W_up.dtype)
        W_down[:, TARGET_I] = (alpha   * v_meow_n.to(DEVICE)).to(W_down.dtype)

        print(f"\n{'='*60}\n  ALPHA = {alpha}  (SCALE_G={SCALE_G}, SCALE_U={SCALE_U})\n{'='*60}",
              flush=True)
        results[alpha] = {}
        for name, p in PROBES:
            text, gen_ids = gen(p, max_new=300)
            n_meow = text.count("喵")
            # 简单看"喵后接句号"的次数 (合规模式)
            n_meow_before_period = 0
            for i, c in enumerate(text):
                if c == "喵" and i + 1 < len(text) and text[i + 1] in ("。", ".", "!", "?"):
                    n_meow_before_period += 1
            results[alpha][name] = {
                "text": text, "n_meow": n_meow,
                "n_meow_before_period": n_meow_before_period,
            }
            tag = f"  [喵×{n_meow}, 喵→句号×{n_meow_before_period}]"
            print(f"\n[{name}] {p!r}{tag}", flush=True)
            print(f"  → {text[:300]}", flush=True)

        # 还原
        W_gate[TARGET_I, :] = old_gate
        W_up[TARGET_I, :] = old_up
        W_down[:, TARGET_I] = old_down

    # save
    OUT = Path("J:/amr/amr_wtf/identity_swap/step32_meow_handcraft.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 32: 手搓猫娘神经元 v2 ================\n\n")
        f.write(f"target: L{TARGET_L}#{TARGET_I}\n")
        f.write(f"SCALE_G={SCALE_G}, SCALE_U={SCALE_U}\n")
        f.write(f"positive samples: {len(positive_h)}, negative samples: {len(negative_h)}\n")
        f.write(f"v_disc judge power: pos cos={pos_proj_disc.mean():.3f}, "
                f"neg cos={neg_proj_disc.mean():.3f}, "
                f"gap={(pos_proj_disc.mean()-neg_proj_disc.mean()).item():.3f}\n\n")

        f.write(f"\n--- baseline ---\n")
        for name, p in PROBES:
            text, n = baseline_results[name]
            f.write(f"\n[{name}] {p!r}  [喵×{n}]\n{text}\n")

        for alpha in ALPHA_SWEEP:
            f.write(f"\n\n=== ALPHA = {alpha} ===\n")
            for name, p in PROBES:
                r = results[alpha][name]
                f.write(f"\n[{name}] {p!r}  [喵×{r['n_meow']}, 喵→句号×{r['n_meow_before_period']}]\n{r['text']}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
