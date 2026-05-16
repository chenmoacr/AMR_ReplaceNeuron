"""
Phase 21 step 1: 为 1000 个 inert candidate 提取 W_gate / W_down vocab projection,
准备喂 Gemini 做 'repurpose 安全性' 标注.

输入: phase20_inert_candidates.json (1000 个 {layer, index, cot_max, resp_max, ...})
输出: phase21_inert_for_gemini.json (1000 个 entry, 每个含 vocab projection)

每个 entry:
  id: "L##I"
  layer / index / cot_max / resp_max / combined_max / cot_mean / resp_mean
  gate_vocab_top_push : top-15 ASCII tokens W_gate row @ lm_head 正向投影
  gate_vocab_top_suppress: top-15 反向
  down_vocab_top_push : top-15 W_down col @ lm_head 正向 (神经元激活后会推这些)
  down_vocab_top_suppress: top-15 反向 (激活后会压这些)
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
CAND_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase20_inert_candidates.json"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase21_inert_for_gemini.json"

DEVICE = "cuda:0"
TOP_VOCAB = 15


def is_ascii_clean(s):
    if not s or not s.strip():
        return False
    return all(c.isascii() and (c.isalnum() or c in " /_.,-+()[]<>{}*#$%@!?:;'\"\\=^~`|") for c in s)


def top_ascii_tokens(proj, tok, k, descending=True):
    order = proj.argsort(descending=descending)
    out = []
    for ti in order.tolist():
        s = tok.decode([ti], skip_special_tokens=False)
        if is_ascii_clean(s) and len(s.strip()) <= 20:
            out.append({"proj": float(proj[ti].item()), "tok": s})
            if len(out) >= k:
                break
    return out


def main():
    print(f"[load] candidates from {CAND_PATH}")
    cands = json.loads(CAND_PATH.read_text(encoding="utf-8"))
    print(f"  {len(cands)} candidates")

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
    print(f"  lm_head shape: {tuple(lm_head_W.shape)}")

    out = []
    import time
    t0 = time.time()
    for rank, c in enumerate(cands, 1):
        L, I = int(c["layer"]), int(c["index"])
        # W_gate row = layers[L].mlp.gate_proj.weight[I, :] 是该神经元的 gate 检测方向
        W_gate_row = layers[L].mlp.gate_proj.weight.data[I, :].to(torch.float32).cpu()
        W_down_col = layers[L].mlp.down_proj.weight.data[:, I].to(torch.float32).cpu()

        gate_proj_vocab = W_gate_row @ lm_head_W.T   # [vocab]
        down_proj_vocab = W_down_col @ lm_head_W.T

        entry = {
            "rank": rank,
            "id": f"L{L:02d}#{I}",
            "layer": L, "index": I,
            "cot_max": c["cot_max"], "cot_mean": c["cot_mean"],
            "resp_max": c["resp_max"], "resp_mean": c["resp_mean"],
            "combined_max": c["combined_max"],
            "W_gate_row_norm": float(W_gate_row.norm().item()),
            "W_down_col_norm": float(W_down_col.norm().item()),
            "gate_vocab_top_push": top_ascii_tokens(gate_proj_vocab, tok, TOP_VOCAB, True),
            "gate_vocab_top_suppress": top_ascii_tokens(gate_proj_vocab, tok, TOP_VOCAB, False),
            "down_vocab_top_push": top_ascii_tokens(down_proj_vocab, tok, TOP_VOCAB, True),
            "down_vocab_top_suppress": top_ascii_tokens(down_proj_vocab, tok, TOP_VOCAB, False),
        }
        out.append(entry)
        if rank <= 3 or rank % 100 == 0:
            elapsed = time.time() - t0
            rate = rank / elapsed
            eta = (len(cands) - rank) / rate
            print(f"  #{rank:>4}  L{L:02d}#{I:<6}  cot_max={c['cot_max']:.4f}  "
                  f"||W_gate||={entry['W_gate_row_norm']:.2f}  ||W_down||={entry['W_down_col_norm']:.2f}  "
                  f"gate_push[:2]={[x['tok'] for x in entry['gate_vocab_top_push'][:2]]}  "
                  f"down_push[:2]={[x['tok'] for x in entry['down_vocab_top_push'][:2]]}  "
                  f"(eta {eta:.0f}s)")

    OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}  ({OUT_PATH.stat().st_size//1024} KB)")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
