"""
Step 4: 干预验证 (causal evidence)
  目标：验证 step 3 找到的 L4#5097 / L4#2713 是否真的承载 Google/Apple 知识

方法：
  teacher-force R 序列（含 Google DeepMind），在 L4 的 mlp.down_proj 输入
  上对候选神经元做 clamp（替换为 R'(Apple) 路径的值）。看 pos 7 的
  next-token logit 是否被推向 R' 轨迹（top1 从 ' Deep' → ' AI'）。

5 个 forward：
  1. R    baseline (no clamp)         → expect logit@pos7 top1 = ' Deep'
  2. R + clamp pair  L4#5097 + L4#2713
  3. R + clamp single L4#5097
  4. R + clamp single L4#2713
  5. R + clamp top-10 of L4 (by step 3 |diff| at pos 7)
  6. R'   baseline (no clamp)         → expect logit@pos7 top1 = ' AI'

对每个 case：
  - 抓 pos 6 / 7 / 8 / 9 的 next-token logit
  - top-10 token + prob
  - 关键 token 的 logit / prob（' Deep'=22267, ' AI'=12498, 'Mind'=65153,
    ' Lab'=10201, ' 开发'=183096）

Output:
  identity_swap/step4_intervene_report.txt
  identity_swap/step4_intervene_raw.pt
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

# R / R' 来自 step 3
R_GEN_IDS = [
    44889, 147224, 236743, 236812, 236900, 5095, 237852,
    6475, 22267, 65153,
    183096, 236918, 63332, 137090, 29854, 237731, 37557, 26609, 236924,
]
REPLACE_POS = (7, 8, 9)
R_PRIME_REPLACE = [9947, 12498, 10201]   # ' Apple AI Lab'

# 干预目标层
TARGET_LAYER = 4

# 关心的 token id
TOK_IDS = {
    " Google":   6475,
    " Apple":    9947,
    " Deep":     22267,
    " AI":       12498,
    "Mind":      65153,
    " Lab":      10201,
    " 开发":      183096,
    " Research": 6498,
    " Inc":      5295,
}


def get_eos_ids(tokenizer):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return list({i for i in eos if i is not None})


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
    print(f"  layers={len(layers)}  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB",
          flush=True)

    # ---- load step 3 raw ----
    raw = torch.load(RAW_PT, weights_only=False)
    R_di = raw["R_down_in"]            # list per-layer [n_pos=10, intermediate]
    Rp_di = raw["Rp_down_in"]
    cfg = raw["config"]
    gen_pos_oi = cfg["gen_pos_of_interest"]   # [5..14]
    abs_pos = cfg["abs_pos"]
    assistant_start = cfg["assistant_start"]
    print(f"[step3] gen_pos_of_interest={gen_pos_oi}  abs_pos={abs_pos}", flush=True)

    # 找 pos 7 在 stored 数组里的索引
    POS_IDX_7 = gen_pos_oi.index(7)
    abs_pos_7 = assistant_start + 7
    print(f"[clamp] target gen-pos=7  abs_pos={abs_pos_7}  POS_IDX_7={POS_IDX_7}",
          flush=True)

    # 抓 R'(Apple) 路径的 L4 在 pos 7 的 intermediate（用作 clamp source）
    L4_Rp_pos7 = Rp_di[TARGET_LAYER][POS_IDX_7].to(DEVICE).bfloat16()  # [intermediate=6144]
    L4_R_pos7  = R_di[TARGET_LAYER][POS_IDX_7].to(DEVICE).bfloat16()
    print(f"[clamp] L4 R  pos7  shape={tuple(L4_R_pos7.shape)}  norm={L4_R_pos7.float().norm().item():.2f}",
          flush=True)
    print(f"[clamp] L4 R' pos7  shape={tuple(L4_Rp_pos7.shape)}  norm={L4_Rp_pos7.float().norm().item():.2f}",
          flush=True)

    # 候选神经元
    diff_L4 = (Rp_di[TARGET_LAYER] - R_di[TARGET_LAYER])[POS_IDX_7]  # [6144]
    top10 = diff_L4.abs().topk(10).indices.tolist()
    print(f"[clamp] L4 top-10 by |diff| at pos7: {top10}", flush=True)

    NEURONS_PAIR = [5097, 2713]
    NEURONS_5097 = [5097]
    NEURONS_2713 = [2713]
    NEURONS_TOP10 = top10

    # ---- build R / R' sequences ----
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

    # ---- forward & capture logits ----
    def forward_with_clamp(seq, clamp_target_pos=None, clamp_neurons=None,
                            clamp_source=None):
        """
        Run forward with optional clamp on TARGET_LAYER.mlp.down_proj input
        at clamp_target_pos. Replace the indexed neurons with values from
        clamp_source (a [intermediate] tensor on device).
        Returns logits [T, vocab] (only positions of interest used).
        """
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

        handle = None
        if clamp_neurons is not None:
            neuron_idx = torch.tensor(clamp_neurons, dtype=torch.long, device=DEVICE)
            tgt_pos = clamp_target_pos
            src = clamp_source

            def pre(module, inputs):
                x = inputs[0]   # [1, T, intermediate]
                # 1. clone to avoid in-place on grad-tracked tensor
                x = x.clone()
                # 2. replace at (batch=0, pos=tgt_pos, neuron_idx)
                x[0, tgt_pos, neuron_idx] = src[neuron_idx]
                return (x,) + inputs[1:]

            handle = layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(pre)

        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0].float().cpu()   # [T, vocab]
        finally:
            if handle is not None:
                handle.remove()
        return logits

    # ---- run all cases ----
    cases = [
        ("R_baseline",   "R",  None, None,           None),
        ("R_clamp_pair", "R",  abs_pos_7, NEURONS_PAIR,  L4_Rp_pos7),
        ("R_clamp_5097", "R",  abs_pos_7, NEURONS_5097,  L4_Rp_pos7),
        ("R_clamp_2713", "R",  abs_pos_7, NEURONS_2713,  L4_Rp_pos7),
        ("R_clamp_top10","R",  abs_pos_7, NEURONS_TOP10, L4_Rp_pos7),
        ("Rp_baseline",  "Rp", None, None,           None),
    ]

    results = {}
    for name, seq_choice, tgt_pos, neurons, src in cases:
        seq = seq_R if seq_choice == "R" else seq_Rp
        torch.cuda.empty_cache()
        t0 = time.time()
        logits = forward_with_clamp(seq, tgt_pos, neurons, src)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        torch.cuda.reset_peak_memory_stats(0)
        nstr = "(-)" if neurons is None else f"{len(neurons)}n"
        print(f"\n[{name}]  seq={seq_choice}  clamp={nstr}  "
              f"{dt:.1f}s  peak={peak:.2f}GB", flush=True)
        results[name] = logits

    # ---- analyze ----
    # focus positions: pos 6 (next = " Google"/" Apple" for R/R')
    #                  pos 7 (next = " Deep"/" AI" for R/R')
    #                  pos 8 (next = "Mind"/" Lab" for R/R')
    #                  pos 9 (next = " 开发" for both)
    focus_gen_pos = [6, 7, 8, 9]
    print(f"\n========================================================================")
    print(f"  Next-token logits at gen positions {focus_gen_pos}")
    print(f"========================================================================")

    report_lines = []
    report_lines.append("================ Step 4: 干预验证 ================\n\n")
    report_lines.append(f"target layer: L{TARGET_LAYER}\n")
    report_lines.append(f"clamp source: R'(Apple) path L{TARGET_LAYER} pos 7 intermediate\n")
    report_lines.append(f"L4 top-10 by |diff| at pos 7: {NEURONS_TOP10}\n\n")

    for gp in focus_gen_pos:
        abs_p = assistant_start + gp
        # next-token logit at pos abs_p predicts token at pos abs_p+1
        actual_next_R  = seq_R[abs_p + 1] if abs_p + 1 < len(seq_R) else None
        actual_next_Rp = seq_Rp[abs_p + 1] if abs_p + 1 < len(seq_Rp) else None
        line = (f"\n--- gen-pos {gp}  (abs={abs_p})  "
                f"actual_next R={tok.decode([actual_next_R]) if actual_next_R else '?'!r}  "
                f"R'={tok.decode([actual_next_Rp]) if actual_next_Rp else '?'!r} ---")
        print(line, flush=True)
        report_lines.append(line + "\n")

        # column header
        header = f"  {'case':<14} | {'top1':<14} {'prob':>6} | {'top2':<14} {'prob':>6}"
        for name, tid in TOK_IDS.items():
            header += f" | {repr(name)[1:-1]:>9}"
        print(header, flush=True)
        report_lines.append(header + "\n")

        for name, _ in [(c[0], None) for c in cases]:
            logits = results[name][abs_p]   # [vocab]
            probs = F.softmax(logits, dim=-1)
            top = probs.topk(5)
            top_ids = top.indices.tolist()
            top_probs = top.values.tolist()
            top1_tok = repr(tok.decode([top_ids[0]]))
            top2_tok = repr(tok.decode([top_ids[1]]))
            row = (f"  {name:<14} | {top1_tok[:14]:<14} {top_probs[0]:>6.3f} | "
                   f"{top2_tok[:14]:<14} {top_probs[1]:>6.3f}")
            for tname, tid in TOK_IDS.items():
                row += f" | {probs[tid].item():>9.4f}"
            print(row, flush=True)
            report_lines.append(row + "\n")

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "results": {n: l for n, l in results.items()},
        "config": {
            "target_layer": TARGET_LAYER,
            "neurons_pair": NEURONS_PAIR,
            "neurons_top10": NEURONS_TOP10,
            "clamp_target_abs_pos": abs_pos_7,
            "L4_R_pos7": L4_R_pos7.float().cpu(),
            "L4_Rp_pos7": L4_Rp_pos7.float().cpu(),
        },
    }, OUT_DIR / "step4_intervene_raw.pt")

    with open(OUT_DIR / "step4_intervene_report.txt", "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\n[save] step4_intervene_raw.pt", flush=True)
    print(f"[save] step4_intervene_report.txt", flush=True)
    print(f"\n[done]", flush=True)


if __name__ == "__main__":
    main()
