"""
Phase 6: logit lens 扫 K=30 clamp 30 个神经元 vocab 投影, 看 push/suppress 哪些词.

对每个 (L, neuron_idx, diff_sign):
  vocab_proj = W_down[L][:, neuron_idx] @ lm_head.weight.T  # shape [vocab]
  top-15 positive: neuron act>0 时 push 的 tokens
  top-15 negative: neuron act>0 时 suppress 的 tokens

K=30 clamp gain 符号:
  diff > 0 (opus > base): clamp gain = +1.5 (boost)
  diff < 0 (base > opus): clamp gain = -1.5 (suppress base 多用的 neuron)

如果 user "RLHF 负面产物" 假说成立: diff < 0 的 neurons 应该 push "string DP" 词
("string", "S[", "L =", "Case", "leading digit", "positions $i$", etc.).
clamp gain=-1.5 反向激活 → push 这些词的方向反 → suppress string DP framing.
"""
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase6_logit_lens_K30.txt"

DEVICE = "cuda:0"
TOP_K = 15

# 经验关键词 cluster (用于自动标记)
STRING_DP_KW = ["string", "S[", "L =", "positions", "Case", "leading", "iterate",
                 "prefix", "suffix", "S =", "L_d", "0..L"]
POSITIONAL_DP_KW = ["high", " cur", " low", "P =", " P ", "m =", " m ", "power_of_10",
                     "10^", "digit at", "$D$", "$P$", "$B$", " $A$", "cycle"]


def classify_token(s):
    """Return 'STRING_DP' or 'POSITIONAL' or '-' for the given decoded token string."""
    for kw in STRING_DP_KW:
        if kw in s:
            return "STRING_DP"
    for kw in POSITIONAL_DP_KW:
        if kw in s:
            return "POSITIONAL"
    return "-"


def main():
    print("[load] model...")
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
    lm_head_W = model.lm_head.weight.to(torch.float32).cpu()   # [vocab, hidden=1536]

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"][:30]   # top 30 used by K=30 clamp

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ Phase 6: K=30 clamp 30 神经元 logit lens vocab 投影 ================\n\n")
        f.write("每个 neuron 显示:\n")
        f.write("  diff: phase2 (opus_window - base_window) 平均激活差\n")
        f.write("  clamp gain: ±1.5 (符号同 diff 符号)\n")
        f.write("  W_down 列 vocab top-K push (positive proj, act>0 时 push 的 token)\n")
        f.write("  W_down 列 vocab top-K suppress (negative proj, act>0 时 suppress 的 token)\n")
        f.write("  classify: STRING_DP / POSITIONAL / - (基于关键词匹配)\n\n")
        f.write("="*100 + "\n\n")

        # summary counter
        string_dp_push_count = 0
        positional_push_count = 0

        for rank, (L, n_idx, diff) in enumerate(top_diff, 1):
            W_down_col = layers[L].mlp.down_proj.weight.data[:, n_idx].to(torch.float32).cpu()
            proj = W_down_col @ lm_head_W.T   # [vocab]

            top_pos = torch.topk(proj, TOP_K)
            top_neg = torch.topk(-proj, TOP_K)

            gain = +1.5 if diff > 0 else -1.5
            header = f"#{rank:>2}  L{L:02d}#{n_idx:<6d}  diff={diff:+.3f}  clamp_gain={gain:+.1f}"
            print(f"\n{header}")
            f.write(f"\n{'='*70}\n{header}\n{'='*70}\n")

            f.write(f"\n  TOP {TOP_K} PUSH (act>0 时 logit 升):\n")
            push_tokens_str = []
            for v, ti in zip(top_pos.values.tolist(), top_pos.indices.tolist()):
                s = tok.decode([ti], skip_special_tokens=False)
                cls = classify_token(s)
                f.write(f"    proj={v:+.4f}  [{cls:<10}]  {s!r}\n")
                push_tokens_str.append(s)
                if cls == "STRING_DP":
                    string_dp_push_count += 1
                elif cls == "POSITIONAL":
                    positional_push_count += 1

            f.write(f"\n  TOP {TOP_K} SUPPRESS (act>0 时 logit 降):\n")
            for v, ti in zip(top_neg.values.tolist(), top_neg.indices.tolist()):
                s = tok.decode([ti], skip_special_tokens=False)
                cls = classify_token(s)
                f.write(f"    proj={-v:+.4f}  [{cls:<10}]  {s!r}\n")

            # 简短 stdout summary
            push_summary = " | ".join(repr(s)[:15] for s in push_tokens_str[:5])
            print(f"  push top5: {push_summary}")

        f.write(f"\n\n================ Aggregate ================\n")
        f.write(f"Across all {TOP_K * 30} top-push token positions:\n")
        f.write(f"  STRING_DP tokens hit:  {string_dp_push_count}\n")
        f.write(f"  POSITIONAL tokens hit: {positional_push_count}\n")

    print(f"\n[save] {OUT_PATH}")
    print(f"[aggregate] STRING_DP push hits: {string_dp_push_count}, POSITIONAL: {positional_push_count}")


if __name__ == "__main__":
    main()
