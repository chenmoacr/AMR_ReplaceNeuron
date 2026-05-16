"""
Phase 5b (revised per Opus): single-forward α sweep — 不每个 α 都 6000 token gen.

prefix = phase4a cot 直到 "Contribution is " (D>1 错位点)
对每个 α: edit W_down 列 → single forward → 抓 last token logit
看 logit('0') vs logit('$') 变化, 找 logit_diff 翻转 (从 baseline +27.74 → 负值) 的 α
candidate α 范围更细 [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0]

也加 K=30 clamp 作为 framing 层 baseline (因为 5a 验证单 edit 不能 flip framing).

实测每 α 只 ~10 sec forward, 比 6000-token gen 快 ~30x.
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
TARGET_L = 33
TARGET_I = 11786
ALPHAS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0]


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

    W_down = layers[TARGET_L].mlp.down_proj.weight.data
    col_orig = W_down[:, TARGET_I].detach().clone()
    proj_orig = (col_orig.to(torch.float32).cpu() @ e_diff_n).item()
    print(f"\n[L{TARGET_L}#{TARGET_I}] proj_orig={proj_orig:+.4f}")

    # tokens we care about
    cand_ids = {
        "0":    zero_id,
        "$":    dollar_id,
        "P":    tok.encode("P", add_special_tokens=False)[0],
        " $P$": tok.encode(" $P$", add_special_tokens=False)[0],
        " 0":   tok.encode(" 0", add_special_tokens=False)[0],
    }
    print(f"[cands] {cand_ids}")

    def fwd_logit():
        hooks = install_clamp_hooks(layers, sel_clamp)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=full_ids, use_cache=False)
        finally:
            remove_hooks(hooks)
        return out.logits[0, -1, :].to(torch.float32).cpu()

    print(f"\n{'='*70}")
    print(f"{'α':>5} {'proj_new':>9}  {'logit_0':>8} {'logit_$':>8} {'logit_P':>8} {'l_0-l_$':>8} {'top-1':>20}")
    print('-'*70)
    results = []
    for alpha in ALPHAS:
        W_down[:, TARGET_I] = col_orig
        if alpha > 0:
            col = col_orig.to(torch.float32).cpu()
            proj = (col @ e_diff_n).item()
            col_new = col - (1.0 + alpha) * proj * e_diff_n
            W_down[:, TARGET_I] = col_new.to(W_down.dtype).to(W_down.device)
            new_proj = (col_new @ e_diff_n).item()
        else:
            new_proj = proj_orig
        logits = fwd_logit()
        l0 = logits[cand_ids["0"]].item()
        ld = logits[cand_ids["$"]].item()
        lp = logits[cand_ids["P"]].item()
        top1_id = logits.argmax().item()
        top1_str = tok.decode([top1_id], skip_special_tokens=False)
        print(f" {alpha:>4.1f} {new_proj:+8.4f}  {l0:+8.3f} {ld:+8.3f} {lp:+8.3f} {l0-ld:+8.3f}  {top1_str!r:>20}")
        results.append({
            "alpha": alpha, "proj": new_proj,
            "l0": l0, "l_dollar": ld, "l_P": lp,
            "diff_0_dollar": l0 - ld,
            "top1_str": top1_str, "top1_id": top1_id,
        })
    W_down[:, TARGET_I] = col_orig

    # save
    save_path = OUT_DIR / "phase5b_single_fwd_sweep.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5b single-forward sweep ================\n\n")
        f.write(f"K_clamp={K} ABS_CLAMP={ABS_CLAMP}, edit L{TARGET_L}#{TARGET_I}\n")
        f.write(f"proj_orig={proj_orig:+.4f}\n\n")
        f.write(f"{'α':>5} {'proj_new':>9}  {'l_0':>8} {'l_$':>8} {'l_P':>8} {'l_0-l_$':>8}  top1\n")
        for r in results:
            f.write(f" {r['alpha']:>4.1f} {r['proj']:+8.4f}  {r['l0']:+8.3f} {r['l_dollar']:+8.3f} {r['l_P']:+8.3f} {r['diff_0_dollar']:+8.3f}  {r['top1_str']!r}\n")
    print(f"\n[save] {save_path}")

    # identify promising α: where diff_0_dollar flipped or top1 changed
    flipped = [r for r in results if r["diff_0_dollar"] < 0 or r["top1_str"] != "'0'"]
    print(f"\n[promising α (logit_diff < 0 or top1 != '0')]: {len(flipped)}")
    for r in flipped:
        print(f"  α={r['alpha']}  diff={r['diff_0_dollar']:+.3f}  top1={r['top1_str']!r}")


if __name__ == "__main__":
    main()
