"""
Phase 10 step 2: 准备 30 个 K=30 neurons 数据成 JSON (准备喂 Gemini 标注).

每个 neuron 提供:
  - id (L##I)
  - layer / index
  - phase2_diff (base-opus 平均激活差)
  - clamp_gain (K=30 clamp 实际用的 gain = sign(diff)*1.5)
  - gate_vocab_top_push: W_gate column @ lm_head.T 前 15 ASCII token (= gate 'reads' 什么 concept)
  - gate_vocab_top_suppress: W_gate column 反向 top 15 ASCII
  - down_vocab_top_push: W_down column @ lm_head.T 前 15 ASCII (= neuron 'pushes' 什么)
  - down_vocab_top_suppress: W_down column 反向 top 15 ASCII (suppress)
  - 多语言 token 用 [foreign] 占位, 只输出 ASCII clean for Gemini consumption clarity
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase10_neurons_for_gemini.json"

DEVICE = "cuda:0"
N_NEURONS = 30
TOP_VOCAB = 15


def is_ascii_clean(s):
    if not s or not s.strip():
        return False
    return all(c.isascii() and (c.isalnum() or c in " /_.,-+()[]<>{}*#$%@!?:;'\"\\=^~`|") for c in s)


def top_ascii_tokens(proj, tok, k, descending=True):
    """Returns list of (proj_value, token_string) for top-k ASCII tokens."""
    order = proj.argsort(descending=descending)
    out = []
    for ti in order.tolist():
        s = tok.decode([ti], skip_special_tokens=False)
        if is_ascii_clean(s):
            disp = s
            if len(disp.strip()) <= 20:
                out.append({"proj": float(proj[ti].item()), "tok": disp})
                if len(out) >= k:
                    break
    return out


def main():
    print("[load] tokenizer + model (BF16)...")
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
    lm_head_W = model.lm_head.weight.to(torch.float32).cpu()

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"][:N_NEURONS]
    print(f"[loaded] {len(top_diff)} K={N_NEURONS} clamp neurons")

    out = []
    for rank, (L, I, diff) in enumerate(top_diff, 1):
        L, I, diff = int(L), int(I), float(diff)
        W_gate_col = layers[L].mlp.gate_proj.weight.data[I, :].to(torch.float32).cpu()
        W_down_col = layers[L].mlp.down_proj.weight.data[:, I].to(torch.float32).cpu()
        gate_proj_vocab = W_gate_col @ lm_head_W.T
        down_proj_vocab = W_down_col @ lm_head_W.T

        gate_push = top_ascii_tokens(gate_proj_vocab, tok, TOP_VOCAB, True)
        gate_suppress = top_ascii_tokens(gate_proj_vocab, tok, TOP_VOCAB, False)
        down_push = top_ascii_tokens(down_proj_vocab, tok, TOP_VOCAB, True)
        down_suppress = top_ascii_tokens(down_proj_vocab, tok, TOP_VOCAB, False)

        clamp_gain = 1.5 if diff > 0 else -1.5
        entry = {
            "rank": rank,
            "id": f"L{L:02d}#{I}",
            "layer": L,
            "index": I,
            "phase2_diff": diff,
            "clamp_gain": clamp_gain,
            "interpretation_hint": (
                "diff>0 means this neuron fires more in 'opus path' (correct positional DP cot); "
                "diff<0 means it fires more in 'base path' (broken string DP cot). "
                "K=30 clamp uses gain=sign(diff)*1.5 to invert the activation. "
                "W_gate column projected onto lm_head approximates 'what token concept the gate reads as detection signal'. "
                "W_down column projected onto lm_head approximates 'what tokens this neuron pushes (positive proj) or suppresses (negative proj)' when activated."
            ),
            "gate_vocab_top_push": gate_push,
            "gate_vocab_top_suppress": gate_suppress,
            "down_vocab_top_push": down_push,
            "down_vocab_top_suppress": down_suppress,
        }
        out.append(entry)
        print(f"  #{rank:>2}  L{L:02d}#{I:<6d} diff={diff:+.3f}  "
              f"gate_push[:3]={[x['tok'] for x in gate_push[:3]]}  "
              f"down_push[:3]={[x['tok'] for x in down_push[:3]]}")

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}  ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
