"""
Step 4b: 加强版干预 — 探索"扰动多大才能动 logit"

step 4 阴性（单/双/top-10 神经元 clamp 无效）。本步加强干预：
  A. clamp L4 整个 mlp.down_proj 输入（6144→6144 整体替换为 R' 路径）
  B. clamp 多层 top-10：L0/L1/L4/L25/L34 同时 clamp top-10 each
  C. clamp 全部层 top-30：所有 35 层各自 top-30 神经元 clamp
  D. clamp L4 整个 mlp 输入 + L34 整个 mlp 输入（双层 full swap）
  E. layer-by-layer full swap：每一层单独 swap 整个 L_i mlp 输入，
     看哪一层 swap 后 logit 最接近 R' (即"哪一层是决策层")

Output: identity_swap/step4b_strong_report.txt
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
RAW_PT = OUT_DIR / "step3_diff_raw.pt"
DEVICE = "cuda:0"
PROMPT_TEXT = "你是谁?"

R_GEN_IDS = [
    44889, 147224, 236743, 236812, 236900, 5095, 237852,
    6475, 22267, 65153,
    183096, 236918, 63332, 137090, 29854, 237731, 37557, 26609, 236924,
]
REPLACE_POS = (7, 8, 9)
R_PRIME_REPLACE = [9947, 12498, 10201]

TOK_IDS = {
    " Deep": 22267, " AI": 12498, " Lab": 10201, "Mind": 65153,
    " 开发": 183096, " Research": 6498, " 研究": 106774,
    " Google": 6475, " Apple": 9947,
}


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
    layers = model.model.language_model.layers
    n_layers = len(layers)
    print(f"  layers={n_layers}  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB",
          flush=True)

    raw = torch.load(RAW_PT, weights_only=False)
    R_di = raw["R_down_in"]
    Rp_di = raw["Rp_down_in"]
    cfg = raw["config"]
    gen_pos_oi = cfg["gen_pos_of_interest"]
    assistant_start = cfg["assistant_start"]
    POS_IDX_7 = gen_pos_oi.index(7)
    abs_pos_7 = assistant_start + 7

    # ---- precompute per-layer R' values at pos 7 (clamp source) ----
    Rp_pos7_per_layer = [Rp_di[L][POS_IDX_7].to(DEVICE).bfloat16() for L in range(n_layers)]
    R_pos7_per_layer  = [R_di[L][POS_IDX_7].to(DEVICE).bfloat16()  for L in range(n_layers)]

    # ---- top neurons per layer at pos 7 ----
    diff_per_layer = [(Rp_di[L] - R_di[L])[POS_IDX_7] for L in range(n_layers)]
    top30_per_layer = [d.abs().topk(30).indices.tolist() for d in diff_per_layer]

    # ---- build sequences ----
    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq_R = list(prefix_ids) + list(R_GEN_IDS)
    seq_Rp = list(prefix_ids) + list(R_GEN_IDS)
    for i, p in enumerate(REPLACE_POS):
        seq_Rp[assistant_start + p] = R_PRIME_REPLACE[i]

    # ---- forward with multi-layer clamp ----
    def forward_with_clamps(seq, clamp_specs):
        """
        clamp_specs : list of (layer_idx, neuron_idx_list_or_None, source_tensor)
        If neuron_idx_list_or_None is None → replace ALL neurons at pos 7
        """
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        handles = []
        for L, neurons, src in clamp_specs:
            if neurons is None:
                # full swap
                def make_pre_full(s):
                    def pre(module, inputs):
                        x = inputs[0].clone()
                        x[0, abs_pos_7, :] = s
                        return (x,) + inputs[1:]
                    return pre
                handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(
                    make_pre_full(src)))
            else:
                neuron_idx = torch.tensor(neurons, dtype=torch.long, device=DEVICE)
                def make_pre_sub(s, n_idx):
                    def pre(module, inputs):
                        x = inputs[0].clone()
                        x[0, abs_pos_7, n_idx] = s[n_idx]
                        return (x,) + inputs[1:]
                    return pre
                handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(
                    make_pre_sub(src, neuron_idx)))
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0].float().cpu()
        finally:
            for h in handles:
                h.remove()
        return logits

    # ---- cases ----
    print(f"\n========================================================================")
    print(f"  Strong intervention sweep")
    print(f"========================================================================")

    cases = []
    cases.append(("R_baseline", "R", []))
    cases.append(("Rp_baseline", "Rp", []))

    # A. L4 full swap
    cases.append(("L4_full_swap", "R", [(4, None, Rp_pos7_per_layer[4])]))
    # B. multi-layer top-10 (focal layers)
    focal = [0, 1, 4, 25, 34]
    multi_top10 = [(L, top30_per_layer[L][:10], Rp_pos7_per_layer[L]) for L in focal]
    cases.append((f"L{focal}_top10", "R", multi_top10))
    # C. all layers top-30
    all_top30 = [(L, top30_per_layer[L], Rp_pos7_per_layer[L]) for L in range(n_layers)]
    cases.append(("all_layers_top30", "R", all_top30))
    # D. L4 + L34 full swap
    cases.append(("L4_L34_full", "R",
                  [(4, None, Rp_pos7_per_layer[4]),
                   (34, None, Rp_pos7_per_layer[34])]))
    # E. all layers full swap
    all_full = [(L, None, Rp_pos7_per_layer[L]) for L in range(n_layers)]
    cases.append(("all_layers_full", "R", all_full))

    results = {}
    for name, seq_choice, clamp_specs in cases:
        seq = seq_R if seq_choice == "R" else seq_Rp
        torch.cuda.empty_cache()
        t0 = time.time()
        logits = forward_with_clamps(seq, clamp_specs)
        dt = time.time() - t0
        torch.cuda.reset_peak_memory_stats(0)
        print(f"\n[{name}]  seq={seq_choice}  n_clamps={len(clamp_specs)}  {dt:.1f}s",
              flush=True)
        results[name] = logits

    # ---- analyze ----
    abs_p_pos7 = assistant_start + 7
    print(f"\n--- Next-token logit at gen-pos 7 (true R next=' Deep', R' next=' AI') ---")
    print(f"  {'case':<22} | {'top1':<14} {'prob':>6} | {'top3':<35} | "
          f"{'Deep':>6} {'AI':>6} {'开发':>6} {'Research':>8} {'Lab':>6}")
    report_lines = ["================ Step 4b: 加强干预 ================\n\n"]

    for name, _, _ in cases:
        logits = results[name][abs_p_pos7]
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(5)
        top_ids = top.indices.tolist()
        top_probs = top.values.tolist()
        top1 = repr(tok.decode([top_ids[0]]))
        top3_str = " / ".join(f"{repr(tok.decode([i]))}={p:.3f}"
                              for i, p in zip(top_ids[:3], top_probs[:3]))
        line = (f"  {name:<22} | {top1[:14]:<14} {top_probs[0]:>6.3f} | {top3_str:<35} | "
                f"{probs[TOK_IDS[' Deep']]:>6.3f} {probs[TOK_IDS[' AI']]:>6.3f} "
                f"{probs[TOK_IDS[' 开发']]:>6.3f} {probs[TOK_IDS[' Research']]:>8.3f} "
                f"{probs[TOK_IDS[' Lab']]:>6.3f}")
        print(line, flush=True)
        report_lines.append(line + "\n")

    # ---- F. layer-by-layer full swap (which layer is "the decision layer") ----
    print(f"\n========================================================================")
    print(f"  F. Layer-by-layer full swap (find the critical layer)")
    print(f"========================================================================")
    print(f"  Each row: clamp ONE layer's full mlp.down_proj input at pos 7 to R'(Apple) path")
    print(f"  {'L':>3} | {'top1':<14} {'prob':>6} | {'Deep':>6} {'AI':>6} {'开发':>6} {'Research':>8}")
    report_lines.append("\n--- F. Layer-by-layer full swap ---\n")
    layer_results = []
    for L in range(n_layers):
        torch.cuda.empty_cache()
        logits = forward_with_clamps(seq_R,
                                      [(L, None, Rp_pos7_per_layer[L])])
        logp7 = logits[abs_p_pos7]
        probs = F.softmax(logp7, dim=-1)
        top = probs.topk(3)
        top_ids = top.indices.tolist()
        top_probs = top.values.tolist()
        top1 = repr(tok.decode([top_ids[0]]))
        line = (f"  L{L:>2} | {top1[:14]:<14} {top_probs[0]:>6.3f} | "
                f"{probs[TOK_IDS[' Deep']]:>6.3f} {probs[TOK_IDS[' AI']]:>6.3f} "
                f"{probs[TOK_IDS[' 开发']]:>6.3f} {probs[TOK_IDS[' Research']]:>8.3f}")
        print(line, flush=True)
        report_lines.append(line + "\n")
        layer_results.append({
            "L": L, "top1": top1, "p_top1": top_probs[0],
            "p_deep": probs[TOK_IDS[" Deep"]].item(),
            "p_ai":   probs[TOK_IDS[" AI"]].item(),
            "p_kaifa": probs[TOK_IDS[" 开发"]].item(),
            "p_research": probs[TOK_IDS[" Research"]].item(),
        })

    # rank by p_ai
    sorted_by_ai = sorted(layer_results, key=lambda r: r["p_ai"], reverse=True)
    print(f"\n  → Top 5 layers by p(' AI') after clamp:")
    report_lines.append("\n  Top 5 layers by p(' AI') after full swap:\n")
    for r in sorted_by_ai[:5]:
        ln = f"    L{r['L']:>2}  p_AI={r['p_ai']:.4f}  p_Deep={r['p_deep']:.4f}  top1={r['top1']}"
        print(ln)
        report_lines.append(ln + "\n")

    sorted_by_kaifa = sorted(layer_results, key=lambda r: r["p_kaifa"], reverse=True)
    print(f"\n  → Top 5 layers by p(' 开发') after clamp:")
    report_lines.append("\n  Top 5 layers by p(' 开发') after full swap:\n")
    for r in sorted_by_kaifa[:5]:
        ln = f"    L{r['L']:>2}  p_开发={r['p_kaifa']:.4f}  top1={r['top1']}"
        print(ln)
        report_lines.append(ln + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step4b_strong_report.txt", "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    torch.save({
        "cases": list(results.keys()),
        "logits_pos7": {n: l[abs_p_pos7].cpu() for n, l in results.items()},
        "layer_sweep": layer_results,
    }, OUT_DIR / "step4b_strong_raw.pt")
    print(f"\n[save] step4b_strong_report.txt", flush=True)


if __name__ == "__main__":
    main()
