"""
Step 13: 神经元级 logit-diff 归因 — 找出"让 Google 赢 OpenAI 的具体神经元"

回答用户的核心问题：
  在 35 帧 × 1536 维的 residual stream 里
  L25 时 logit lens 看 OpenAI > Google
  L26 之后 Google > OpenAI
  那么 L26 这一帧里，**哪些神经元**在放大 Google 缩小 OpenAI？

方法：
  1. forward, capture per-layer residual / attn_out / mlp_out 在 last_pos 的值
  2. logit_diff(Google - OpenAI) 演化曲线 (per-layer)
  3. per-(layer, component) attribution: 每层 attn/mlp 加多少 logit_diff
  4. per-neuron attribution at L25/L26/L27 mlp
     contrib_neuron_i = activation[i] · (W_down[:,i] @ (W_unembed[Google] - W_unembed[OpenAI]))
  5. 验证: clamp top-K 推 Google neurons → 看 logit_diff 变化

输出：
  step13_logit_diff_attribution.txt
  step13_logit_diff_attribution_raw.pt
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = ROOT / "identity_swap"
DEVICE = "cuda:0"

PROMPT_TEXT = "你是谁?"
PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由
GOOGLE_ID = 6475
OPENAI_ID = 131846
APPLE_ID  = 9947

TOP_K = 30
ATTRIBUTION_LAYERS = [22, 23, 24, 25, 26, 27, 28]   # 关键过渡层
CLAMP_LAYER = 26
CLAMP_TOP_K = 10


def main():
    print(f"[load] {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    text_model = model.model.language_model
    layers = text_model.layers
    final_norm = text_model.norm
    lm_head = model.lm_head
    n_layers = len(layers)

    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + PREFIX_IDS
    last_pos = len(seq) - 1
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    # ---- diff direction in unembed space ----
    W_un = lm_head.weight.float()  # [vocab, hidden]
    diff_dir = (W_un[GOOGLE_ID] - W_un[OPENAI_ID]).cpu()    # [hidden=1536]
    diff_g_a = (W_un[GOOGLE_ID] - W_un[APPLE_ID]).cpu()
    print(f"[unembed] ||W_unembed[Google]||={W_un[GOOGLE_ID].norm().item():.3f}  "
          f"||W_unembed[OpenAI]||={W_un[OPENAI_ID].norm().item():.3f}", flush=True)
    print(f"[diff_dir] ||W[Google] - W[OpenAI]|| = {diff_dir.norm().item():.3f}", flush=True)

    # ---- 1. Forward, capture all activations ----
    captures = {
        "resid_after": [None] * n_layers,
        "resid_before": [None] * n_layers,
        "attn_out":    [None] * n_layers,
        "mlp_out":     [None] * n_layers,
        "mlp_intermediate": [None] * n_layers,
    }
    handles = []
    for L in range(n_layers):
        def make_pre(L=L):
            def pre(m, i):
                captures["resid_before"][L] = i[0][0, last_pos].detach().clone()
            return pre
        handles.append(layers[L].register_forward_pre_hook(make_pre()))

        def make_post(L=L):
            def post(m, i, o):
                x = o[0] if isinstance(o, tuple) else o
                captures["resid_after"][L] = x[0, last_pos].detach().clone()
            return post
        handles.append(layers[L].register_forward_hook(make_post()))

        def make_attn(L=L):
            def hook(m, i, o):
                x = o[0] if isinstance(o, tuple) else o
                captures["attn_out"][L] = x[0, last_pos].detach().clone()
            return hook
        handles.append(layers[L].self_attn.register_forward_hook(make_attn()))

        def make_mlp(L=L):
            def hook(m, i, o):
                captures["mlp_out"][L] = o[0, last_pos].detach().clone()
            return hook
        handles.append(layers[L].mlp.register_forward_hook(make_mlp()))

        def make_mlp_in(L=L):
            def pre(m, i):
                captures["mlp_intermediate"][L] = i[0][0, last_pos].detach().clone()
            return pre
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_mlp_in()))

    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        final_logits = out.logits[0, last_pos].float().cpu()
    finally:
        for h in handles: h.remove()

    p_g = F.softmax(final_logits, dim=-1)[GOOGLE_ID].item()
    p_o = F.softmax(final_logits, dim=-1)[OPENAI_ID].item()
    print(f"[final] p(Google)={p_g:.4f}  p(OpenAI)={p_o:.4f}  "
          f"logit_diff(G-O)={final_logits[GOOGLE_ID]-final_logits[OPENAI_ID]:+.3f}",
          flush=True)

    # ---- 2. residual evolution: logit_diff per layer ----
    print(f"\n{'='*78}\n  Residual stream evolution: logit_diff(Google - OpenAI) per layer\n{'='*78}",
          flush=True)
    print(f"{'L':>3} | {'logit(G)':>10} {'logit(O)':>10} {'diff':>10}  marker", flush=True)
    layer_residual_logits = []
    for L in range(n_layers):
        h = captures["resid_after"][L].float().cpu()
        # 简化: 直接 W_un @ h, 不做 final_norm (DLA 标准简化)
        lg = (h * W_un[GOOGLE_ID].cpu()).sum().item()
        lo = (h * W_un[OPENAI_ID].cpu()).sum().item()
        d = lg - lo
        marker = ""
        if L > 0:
            d_prev = layer_residual_logits[-1]["diff"]
            change = d - d_prev
            if abs(change) > 1.0:
                marker = f"  Δ={change:+.2f}"
                if change > 1.0: marker += "  ★ Google 加速"
                elif change < -1.0: marker += "  ★ OpenAI 加速"
        layer_residual_logits.append({"L": L, "logit_g": lg, "logit_o": lo, "diff": d})
        print(f"  L{L:>2} | {lg:>10.3f} {lo:>10.3f} {d:>10.3f}{marker}", flush=True)

    # ---- 3. Per-(layer, component) contribution ----
    # 每层的 attn 和 mlp 各自加多少到 logit_diff
    print(f"\n{'='*78}\n  Per-layer (attn, mlp) contribution to logit_diff", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'L':>3} | {'attn→diff':>12} {'mlp→diff':>12} | {'cum after attn':>15} "
          f"{'cum after mlp':>15}", flush=True)
    cum_attn = []
    cum_mlp = []
    cum = 0.0
    for L in range(n_layers):
        a = captures["attn_out"][L].float().cpu()
        m = captures["mlp_out"][L].float().cpu()
        c_a = (a * diff_dir).sum().item()
        c_m = (m * diff_dir).sum().item()
        cum_attn.append(c_a)
        cum_mlp.append(c_m)
        marker = ""
        if abs(c_a) > 0.5: marker += f"  attn:{c_a:+.2f}"
        if abs(c_m) > 0.5: marker += f"  mlp:{c_m:+.2f}"
        print(f"  L{L:>2} | {c_a:>+12.4f} {c_m:>+12.4f}{marker}", flush=True)

    # ---- 4. Per-neuron attribution in mlp ----
    print(f"\n{'='*78}\n  Per-neuron attribution in mlp (推 Google vs 推 OpenAI)\n{'='*78}",
          flush=True)
    per_layer_neurons = {}
    for L in ATTRIBUTION_LAYERS:
        intermediate = captures["mlp_intermediate"][L].float().cpu()  # [intermediate_dim]
        W_down = layers[L].mlp.down_proj.weight.float().cpu()         # [hidden, intermediate]

        # 每个 neuron i 的 direction score = W_down[:, i] @ diff_dir
        neuron_dir = W_down.T @ diff_dir   # [intermediate]
        # 实际 contribution = activation[i] * direction_score[i]
        contrib = intermediate * neuron_dir   # [intermediate]
        # mlp 总贡献 (sanity check) = contrib.sum()
        sum_contrib = contrib.sum().item()

        print(f"\n--- L{L} mlp neurons (intermediate dim = {len(intermediate)})  "
              f"sum_contrib={sum_contrib:+.4f} (should ≈ {cum_mlp[L]:+.4f}) ---",
              flush=True)

        # top-K 推 Google (positive contrib)
        top_g_idx = contrib.argsort(descending=True)[:TOP_K]
        # top-K 推 OpenAI (negative contrib)
        top_o_idx = contrib.argsort(descending=False)[:TOP_K]

        rows_g = []
        for rank, i in enumerate(top_g_idx.tolist()):
            rows_g.append({
                "rank": rank+1, "neuron": i,
                "contrib": contrib[i].item(),
                "activation": intermediate[i].item(),
                "dir_score": neuron_dir[i].item(),
            })
        rows_o = []
        for rank, i in enumerate(top_o_idx.tolist()):
            rows_o.append({
                "rank": rank+1, "neuron": i,
                "contrib": contrib[i].item(),
                "activation": intermediate[i].item(),
                "dir_score": neuron_dir[i].item(),
            })

        print(f"  top-10 推 Google:", flush=True)
        for r in rows_g[:10]:
            print(f"    {r['rank']:>2}. L{L}#{r['neuron']:<5d}  "
                  f"contrib={r['contrib']:>+8.4f}  "
                  f"act={r['activation']:>+7.3f}  dir={r['dir_score']:>+8.4f}",
                  flush=True)
        print(f"  top-10 推 OpenAI:", flush=True)
        for r in rows_o[:10]:
            print(f"    {r['rank']:>2}. L{L}#{r['neuron']:<5d}  "
                  f"contrib={r['contrib']:>+8.4f}  "
                  f"act={r['activation']:>+7.3f}  dir={r['dir_score']:>+8.4f}",
                  flush=True)

        per_layer_neurons[L] = {
            "top_google": rows_g, "top_openai": rows_o,
            "sum_contrib": sum_contrib,
        }

    # ---- 5. 干预验证：clamp top neurons in L26 ----
    print(f"\n{'='*78}\n  干预验证: clamp L{CLAMP_LAYER} 神经元看 logit_diff 怎么变\n{'='*78}",
          flush=True)
    L = CLAMP_LAYER
    top_g_neurons = [r["neuron"] for r in per_layer_neurons[L]["top_google"][:CLAMP_TOP_K]]
    top_o_neurons = [r["neuron"] for r in per_layer_neurons[L]["top_openai"][:CLAMP_TOP_K]]
    print(f"  top-{CLAMP_TOP_K} 推 Google neurons in L{L}: {top_g_neurons}", flush=True)
    print(f"  top-{CLAMP_TOP_K} 推 OpenAI neurons in L{L}: {top_o_neurons}", flush=True)

    def run_with_clamp(neuron_ids_to_zero):
        ids_idx = torch.tensor(neuron_ids_to_zero, dtype=torch.long, device=DEVICE)
        def pre(m, i):
            x = i[0].clone()
            x[0, last_pos, ids_idx] = 0
            return (x,) + i[1:]
        h = layers[L].mlp.down_proj.register_forward_pre_hook(pre)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0, last_pos].float().cpu()
            probs = F.softmax(logits, dim=-1)
            return logits[GOOGLE_ID].item(), logits[OPENAI_ID].item(), \
                   probs[GOOGLE_ID].item(), probs[OPENAI_ID].item()
        finally:
            h.remove()

    # baseline (no clamp)
    bl_g_lo = final_logits[GOOGLE_ID].item()
    bl_o_lo = final_logits[OPENAI_ID].item()
    bl_g_p = F.softmax(final_logits, dim=-1)[GOOGLE_ID].item()
    bl_o_p = F.softmax(final_logits, dim=-1)[OPENAI_ID].item()
    print(f"\n  baseline:                   logit(G)={bl_g_lo:+.3f}  logit(O)={bl_o_lo:+.3f}  "
          f"diff={bl_g_lo-bl_o_lo:+.3f}  p(G)={bl_g_p:.4f}", flush=True)

    # clamp top推 Google neurons
    g_lo, o_lo, g_p, o_p = run_with_clamp(top_g_neurons)
    print(f"  clamp top-{CLAMP_TOP_K} 推Google: logit(G)={g_lo:+.3f}  logit(O)={o_lo:+.3f}  "
          f"diff={g_lo-o_lo:+.3f}  p(G)={g_p:.4f}  Δp(G)={g_p-bl_g_p:+.4f}",
          flush=True)

    # clamp top推 OpenAI neurons
    g_lo, o_lo, g_p, o_p = run_with_clamp(top_o_neurons)
    print(f"  clamp top-{CLAMP_TOP_K} 推OpenAI: logit(G)={g_lo:+.3f}  logit(O)={o_lo:+.3f}  "
          f"diff={g_lo-o_lo:+.3f}  p(G)={g_p:.4f}  Δp(G)={g_p-bl_g_p:+.4f}",
          flush=True)

    # clamp 双方都 (作为对照)
    g_lo, o_lo, g_p, o_p = run_with_clamp(top_g_neurons + top_o_neurons)
    print(f"  clamp 双方各 {CLAMP_TOP_K}:        logit(G)={g_lo:+.3f}  logit(O)={o_lo:+.3f}  "
          f"diff={g_lo-o_lo:+.3f}  p(G)={g_p:.4f}", flush=True)

    # ---- save report ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step13_logit_diff_attribution.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 13: 神经元级 logit-diff 归因 ================\n\n")
        f.write(f"prompt: {PROMPT_TEXT!r}\n")
        f.write(f"seq: '我是 Gemma 4，一个由'  (last_pos={last_pos})\n")
        f.write(f"target: GOOGLE_ID={GOOGLE_ID}, OPENAI_ID={OPENAI_ID}\n")
        f.write(f"final p(Google)={p_g:.4f}  p(OpenAI)={p_o:.4f}\n\n")

        f.write(f"\n=== Residual stream logit_diff(Google - OpenAI) per layer ===\n")
        f.write(f"{'L':>3} | {'logit(G)':>10} {'logit(O)':>10} {'diff':>10}  Δ\n")
        for i, r in enumerate(layer_residual_logits):
            change = ""
            if i > 0:
                d = r["diff"] - layer_residual_logits[i-1]["diff"]
                if abs(d) > 0.5: change = f"  Δ={d:+.2f}"
            f.write(f"  L{r['L']:>2} | {r['logit_g']:>10.3f} {r['logit_o']:>10.3f} "
                    f"{r['diff']:>10.3f}{change}\n")

        f.write(f"\n=== Per-layer (attn, mlp) contribution to logit_diff ===\n")
        f.write(f"{'L':>3} | {'attn→diff':>12} {'mlp→diff':>12}\n")
        for L in range(n_layers):
            f.write(f"  L{L:>2} | {cum_attn[L]:>+12.4f} {cum_mlp[L]:>+12.4f}\n")

        f.write(f"\n\n=== Per-neuron attribution in mlp ===\n")
        for L in ATTRIBUTION_LAYERS:
            f.write(f"\n--- L{L} mlp (intermediate dim={len(captures['mlp_intermediate'][L])}) "
                    f"sum_contrib≈{per_layer_neurons[L]['sum_contrib']:+.4f} ---\n")
            f.write(f"  TOP-{TOP_K} 推 Google 神经元:\n")
            for r in per_layer_neurons[L]["top_google"]:
                f.write(f"    {r['rank']:>2}. L{L}#{r['neuron']:<5d}  "
                        f"contrib={r['contrib']:>+8.4f}  "
                        f"act={r['activation']:>+7.3f}  "
                        f"dir={r['dir_score']:>+8.4f}\n")
            f.write(f"\n  TOP-{TOP_K} 推 OpenAI 神经元:\n")
            for r in per_layer_neurons[L]["top_openai"]:
                f.write(f"    {r['rank']:>2}. L{L}#{r['neuron']:<5d}  "
                        f"contrib={r['contrib']:>+8.4f}  "
                        f"act={r['activation']:>+7.3f}  "
                        f"dir={r['dir_score']:>+8.4f}\n")

        f.write(f"\n=== 干预验证 (clamp L{CLAMP_LAYER} top-{CLAMP_TOP_K} neurons) ===\n")
        f.write(f"  baseline:                logit_diff(G-O)={bl_g_lo-bl_o_lo:+.3f}  p(G)={bl_g_p:.4f}\n")
        # 重新跑一次记录
        g_lo1, o_lo1, g_p1, o_p1 = run_with_clamp(top_g_neurons)
        f.write(f"  clamp 推Google neurons:  logit_diff={g_lo1-o_lo1:+.3f}  p(G)={g_p1:.4f}  "
                f"Δp={g_p1-bl_g_p:+.4f}\n")
        g_lo2, o_lo2, g_p2, o_p2 = run_with_clamp(top_o_neurons)
        f.write(f"  clamp 推OpenAI neurons:  logit_diff={g_lo2-o_lo2:+.3f}  p(G)={g_p2:.4f}  "
                f"Δp={g_p2-bl_g_p:+.4f}\n")

    torch.save({
        "config": {"prompt": PROMPT_TEXT, "seq": seq, "last_pos": last_pos,
                    "google_id": GOOGLE_ID, "openai_id": OPENAI_ID,
                    "attribution_layers": ATTRIBUTION_LAYERS,
                    "clamp_layer": CLAMP_LAYER, "clamp_top_k": CLAMP_TOP_K},
        "final_p_google": p_g, "final_p_openai": p_o,
        "layer_residual_logits": layer_residual_logits,
        "cum_attn": cum_attn, "cum_mlp": cum_mlp,
        "per_layer_neurons": per_layer_neurons,
    }, OUT_DIR / "step13_logit_diff_attribution_raw.pt")

    print(f"\n[save] step13_logit_diff_attribution.txt", flush=True)
    print(f"[save] step13_logit_diff_attribution_raw.pt", flush=True)


if __name__ == "__main__":
    main()
