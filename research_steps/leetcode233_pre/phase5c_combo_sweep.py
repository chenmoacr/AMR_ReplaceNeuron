"""
Phase 5c: 多神经元 + 广 α single-forward sweep.

5b 教训: 单点 L33#11786 即使 α=5 也只让 logit_diff 从 +27.7 降到 +3.9, 不翻转 top-1.
其他 neuron push '0' 总贡献太大, 单点 cancel 不动.

设计:
  - 组合 1: L33#11786 alone (5b baseline 重测)
  - 组合 2: L33#11786 + L32#5342
  - 组合 3: + L26#1768
  - 组合 4: + L24#9795
  - 组合 5: all 5 (含 L23#1730, 测试 Opus 警告的 act 大 neuron 在组合下表现)
  - α ∈ [1.0, 3.0, 5.0, 8.0, 15.0]

candidates tracked: '0', '$', 'P', ' $P$' (id 609)
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5

# 候选 edit targets (按 phase 4c contrib 排序, 排除 L23#1730 因 act 太大)
EDIT_TARGETS = [
    (33, 11786, "L33#11786 cleanest"),    # contrib=+1.41 act=+4.44 proj=+0.317
    (32, 5342,  "L32#5342"),               # +0.85 +2.16 +0.393
    (26, 1768,  "L26#1768"),               # +0.78 -2.92 -0.268
    (24, 9795,  "L24#9795"),               # +0.73 -2.89 -0.251
    (23, 1730,  "L23#1730 large-act"),     # +3.45 -14.69 -0.235 (Opus 警告)
]

ALPHAS = [1.0, 3.0, 5.0, 8.0, 15.0]


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_hooks(layers, selections):
    by_layer = {}
    for it in selections:
        by_layer.setdefault(int(it["layer"]), []).append((int(it["index"]), float(it["gain"])))
    hooks = []
    for li, items in by_layer.items():
        layer_dev = layers[li].mlp.down_proj.weight.device
        idx = torch.tensor([n for n, _ in items], dtype=torch.long, device=layer_dev)
        shift_cpu = torch.tensor([s for _, s in items], dtype=torch.float32)
        cache = {}
        def make_fn(idx_local, shift_cpu_local, cache_local):
            def pre_hook(module, inputs):
                x = inputs[0]
                key = (x.dtype, x.device)
                shift_dev = cache_local.get(key)
                if shift_dev is None:
                    shift_dev = shift_cpu_local.to(dtype=x.dtype, device=x.device)
                    cache_local[key] = shift_dev
                x = x.clone()
                x[:, :, idx_local] = x[:, :, idx_local] + shift_dev
                return (x,) + inputs[1:]
            return pre_hook
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(
            make_fn(idx, shift_cpu, cache)))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = phase4a_full[phase4a_full.find("\n\n")+2:]
    marker = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    idx = phase4a_text.find(marker)
    prefix_text = phase4a_text[:idx + len(marker)]

    print("[load] tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    lm_head = model.lm_head

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    dollar_id = tok.encode("$", add_special_tokens=False)[0]
    e_diff = (lm_head.weight[zero_id] - lm_head.weight[dollar_id]).to(torch.float32).cpu()
    e_diff_n = F.normalize(e_diff, dim=0)

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel_clamp = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": li, "index": n, "gain": gain})

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(prefix_text, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids + body_ids], dtype=torch.long).to(DEVICE)
    print(f"[full_ids] {full_ids.shape[1]}")

    # backup all edit target columns
    backups = {}
    proj_origs = {}
    for (L, I, _) in EDIT_TARGETS:
        col = layers[L].mlp.down_proj.weight.data[:, I].detach().clone()
        backups[(L, I)] = col
        proj_origs[(L, I)] = (col.to(torch.float32).cpu() @ e_diff_n).item()
        print(f"  L{L}#{I}: proj_orig={proj_origs[(L,I)]:+.4f}")

    cand_ids = {
        "0":     zero_id,
        "$":     dollar_id,
        " $P$":  tok.encode(" $P$", add_special_tokens=False)[0],
        "P":     tok.encode("P", add_special_tokens=False)[0],
    }
    print(f"[cands] {cand_ids}")

    def apply_edits(targets_subset, alpha):
        """edit cols in targets_subset with given alpha (rank-one anti-bug)."""
        for (L, I, _) in targets_subset:
            col = backups[(L, I)].to(torch.float32).cpu()
            proj = (col @ e_diff_n).item()
            col_new = col - (1.0 + alpha) * proj * e_diff_n
            layers[L].mlp.down_proj.weight.data[:, I] = col_new.to(
                layers[L].mlp.down_proj.weight.dtype).to(layers[L].mlp.down_proj.weight.device)

    def restore_all():
        for (L, I, _) in EDIT_TARGETS:
            layers[L].mlp.down_proj.weight.data[:, I] = backups[(L, I)]

    def fwd_logit():
        hooks = install_clamp_hooks(layers, sel_clamp)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=full_ids, use_cache=False)
        finally:
            remove_hooks(hooks)
        return out.logits[0, -1, :].to(torch.float32).cpu()

    # combinations
    combos = [
        ("none (K=30 only)", []),
        ("L33#11786", EDIT_TARGETS[:1]),
        ("+L32#5342", EDIT_TARGETS[:2]),
        ("+L26#1768", EDIT_TARGETS[:3]),
        ("+L24#9795", EDIT_TARGETS[:4]),
        ("+L23#1730 (all)", EDIT_TARGETS[:5]),
    ]

    print(f"\n{'='*95}")
    hdr = f"{'combo':<22} {'α':>4}  {'l_0':>7} {'l_$':>7} {'l_$P$':>7} {'l_P':>7} {'l0-l$':>7} {'l0-l$P$':>8} top1"
    print(hdr)
    print('-'*95)

    rows = []
    for combo_name, targets in combos:
        for alpha in ALPHAS:
            restore_all()
            if targets and alpha > 0:
                apply_edits(targets, alpha)
            logits = fwd_logit()
            l0 = logits[cand_ids["0"]].item()
            ld = logits[cand_ids["$"]].item()
            ldp = logits[cand_ids[" $P$"]].item()
            lp = logits[cand_ids["P"]].item()
            top1_id = logits.argmax().item()
            top1_str = tok.decode([top1_id], skip_special_tokens=False)
            print(f"{combo_name:<22} {alpha:>4.1f}  {l0:+7.2f} {ld:+7.2f} {ldp:+7.2f} {lp:+7.2f} {l0-ld:+7.2f} {l0-ldp:+8.2f}  {top1_str!r}")
            rows.append({
                "combo": combo_name, "alpha": alpha,
                "l_0": l0, "l_$": ld, "l_$P$": ldp, "l_P": lp,
                "diff_0_$": l0-ld, "diff_0_$P$": l0-ldp,
                "top1_str": top1_str,
            })
            # if 'none' baseline, only run once (α doesn't matter)
            if combo_name == "none (K=30 only)":
                break
    restore_all()

    save_path = OUT_DIR / "phase5c_combo_sweep.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5c combo sweep ================\n\n")
        f.write(hdr + "\n")
        for r in rows:
            f.write(f"{r['combo']:<22} {r['alpha']:>4.1f}  {r['l_0']:+7.2f} {r['l_$']:+7.2f} {r['l_$P$']:+7.2f} {r['l_P']:+7.2f} {r['diff_0_$']:+7.2f} {r['diff_0_$P$']:+8.2f}  {r['top1_str']!r}\n")
    print(f"\n[save] {save_path}")

    # find first config where top1 != '0'
    flipped = [r for r in rows if r["top1_str"].strip().strip("'") != "0"]
    print(f"\n[flipped configs ({len(flipped)})]:")
    for r in flipped:
        print(f"  {r['combo']:<22} α={r['alpha']} top1={r['top1_str']!r}")


if __name__ == "__main__":
    main()
