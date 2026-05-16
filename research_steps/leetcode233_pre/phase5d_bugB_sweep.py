"""
Phase 5d: Bug B (code 段 'n /= 10' vs 'P *= 10') single-forward α sweep.

prefix = phase4a code 直到 '// 准备下一位\n        ' (model 即将选 'n' vs 'P')
e_diff_B = lm_head['n'] - lm_head['P']

candidates: L34#3988 (Opus 推, act=+2.48 适中 proj=+0.284), L34#9243 (act=-6.50 大 Opus 警告)
α ∈ [1, 3, 5, 8, 15]
也加 K=30 clamp 作为 framing baseline.
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

EDIT_TARGETS = [
    (34, 3988, "L34#3988 clean"),     # contrib=+0.70 act=+2.48 proj=+0.284
    (34, 9243, "L34#9243 large-act"), # +2.48 -6.50 -0.382 Opus 警告
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
    # prefix 到 'n /= 10;' 的 'n' 之前
    bug_marker = "n /= 10;"
    idx = phase4a_text.find(bug_marker)
    prefix_text = phase4a_text[:idx]
    print(f"[prefix tail 100] {prefix_text[-100:]!r}")

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

    n_id = tok.encode("n", add_special_tokens=False)[0]
    P_id = tok.encode("P", add_special_tokens=False)[0]
    e_diff = (lm_head.weight[n_id] - lm_head.weight[P_id]).to(torch.float32).cpu()
    e_diff_n = F.normalize(e_diff, dim=0)
    print(f"[e_diff] 'n' id={n_id}, 'P' id={P_id}, ||e_diff||={e_diff.norm().item():.3f}")

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

    backups = {}
    for (L, I, _) in EDIT_TARGETS:
        col = layers[L].mlp.down_proj.weight.data[:, I].detach().clone()
        backups[(L, I)] = col
        proj = (col.to(torch.float32).cpu() @ e_diff_n).item()
        print(f"  L{L}#{I}: proj_orig={proj:+.4f}")

    def apply_edit(L, I, alpha):
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

    cand_ids = {
        "n":  n_id,
        "P":  P_id,
        "//": tok.encode("//", add_special_tokens=False)[0],
    }
    print(f"[cands] {cand_ids}")

    print(f"\n{'='*85}")
    hdr = f"{'combo':<22} {'α':>4}  {'l_n':>7} {'l_P':>7} {'l_//':>7} {'l_n-l_P':>8}  top1"
    print(hdr)
    print('-'*85)

    rows = []
    # baseline
    restore_all()
    logits = fwd_logit()
    ln = logits[cand_ids["n"]].item()
    lp = logits[cand_ids["P"]].item()
    lsl = logits[cand_ids["//"]].item()
    top1 = tok.decode([logits.argmax().item()], skip_special_tokens=False)
    print(f"{'none (K=30 only)':<22} {1.0:>4.1f}  {ln:+7.2f} {lp:+7.2f} {lsl:+7.2f} {ln-lp:+8.2f}  {top1!r}")
    rows.append({"combo": "none", "alpha": 0, "l_n": ln, "l_P": lp, "l_//": lsl, "diff": ln-lp, "top1": top1})

    for (L, I, name) in EDIT_TARGETS:
        for alpha in ALPHAS:
            restore_all()
            apply_edit(L, I, alpha)
            logits = fwd_logit()
            ln = logits[cand_ids["n"]].item()
            lp = logits[cand_ids["P"]].item()
            lsl = logits[cand_ids["//"]].item()
            top1 = tok.decode([logits.argmax().item()], skip_special_tokens=False)
            print(f"{name:<22} {alpha:>4.1f}  {ln:+7.2f} {lp:+7.2f} {lsl:+7.2f} {ln-lp:+8.2f}  {top1!r}")
            rows.append({"combo": name, "alpha": alpha, "l_n": ln, "l_P": lp, "l_//": lsl, "diff": ln-lp, "top1": top1})
    restore_all()

    save_path = OUT_DIR / "phase5d_bugB_sweep.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5d Bug B single-fwd sweep ================\n\n")
        f.write(hdr + "\n")
        for r in rows:
            f.write(f"{r['combo']:<22} {r['alpha']:>4.1f}  {r['l_n']:+7.2f} {r['l_P']:+7.2f} {r['l_//']:+7.2f} {r['diff']:+8.2f}  {r['top1']!r}\n")
    print(f"\n[save] {save_path}")

    flipped = [r for r in rows if r["top1"].strip().strip("'") != "n"]
    print(f"\n[flipped configs ({len(flipped)})]:")
    for r in flipped:
        print(f"  {r['combo']:<22} α={r['alpha']:.1f} top1={r['top1']!r}")


if __name__ == "__main__":
    main()
