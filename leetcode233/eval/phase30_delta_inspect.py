"""
Phase 30: 打开 phase28b 训出来的 delta, 看 300 神经元学到了什么.

分析:
  1. ||delta_gate||, ||delta_up||, ||delta_down|| 每神经元的范数 (强度)
  2. delta_down col @ lm_head.T → 每神经元 push 哪些 token (语义意义)
  3. delta_gate row @ lm_head.T → 每神经元 detect 哪些 token (触发条件)
  4. 对比 phase25 手工配方:
     - 手工: W_down = β × lm_head[target_token] (顶峰 1 个 token)
     - 训出: 可能是顶峰 1 个 token, 也可能是分布式
  5. 按 |delta_down| 排序找最"卖力"的神经元

输出: 每神经元的画像 + 排行榜
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
DELTAS_PATH = ROOT / "leetcode233" / "weights" / "phase28b_deltas.pt"
OUT_TXT = ROOT / "leetcode233" / "eval" / "phase30_delta_inspect.txt"
OUT_JSON = ROOT / "leetcode233" / "eval" / "phase30_delta_inspect.json"

DEVICE = "cuda:0"
TOP_VOCAB = 8


def is_ascii_clean(s):
    if not s or not s.strip():
        return False
    return all(c.isascii() and (c.isalnum() or c in " /_.,-+()[]<>{}*#$%@!?:;'\"\\=^~`|\n\t") for c in s)


def top_tokens(proj, tok, k, descending=True, only_ascii=False):
    order = proj.argsort(descending=descending)
    out = []
    for ti in order.tolist():
        s = tok.decode([ti], skip_special_tokens=False)
        if only_ascii and not is_ascii_clean(s):
            continue
        if len(s.strip()) <= 30:
            out.append({"proj": float(proj[ti].item()), "tok": s, "id": ti})
            if len(out) >= k:
                break
    return out


def main():
    print("[load] tokenizer + lm_head + deltas")
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
    lm_head_W = model.lm_head.weight.to(torch.float32).cpu()
    print(f"  lm_head shape: {tuple(lm_head_W.shape)}")

    deltas_state = torch.load(DELTAS_PATH, weights_only=False, map_location="cpu")
    pool = deltas_state["_pool"]
    print(f"  pool size: {len(pool)}")

    # group by layer
    layers_with_deltas = defaultdict(list)
    for p in pool:
        layers_with_deltas[p["layer"]].append(p)
    print(f"  layers: {sorted(layers_with_deltas.keys())}")

    # analyze each neuron
    print(f"\n[analyze] computing per-neuron projections + norms...")
    rows = []
    for L in sorted(layers_with_deltas.keys()):
        d_gate = deltas_state[f"L{L}_gate"].to(torch.float32)   # [n_at_L, hidden]
        d_up   = deltas_state[f"L{L}_up"].to(torch.float32)
        d_down = deltas_state[f"L{L}_down"].to(torch.float32)   # [hidden, n_at_L]
        n_at_L = d_gate.shape[0]
        layer_pool = layers_with_deltas[L]

        for i in range(n_at_L):
            gate_row = d_gate[i]    # [hidden]
            up_row = d_up[i]        # [hidden]
            down_col = d_down[:, i] # [hidden]
            gate_norm = float(gate_row.norm().item())
            up_norm = float(up_row.norm().item())
            down_norm = float(down_col.norm().item())
            # projections
            gate_proj_vocab = gate_row @ lm_head_W.T   # [vocab]
            down_proj_vocab = down_col @ lm_head_W.T

            # 顶峰 1 个 token + 其他 top
            gate_top = top_tokens(gate_proj_vocab, tok, TOP_VOCAB, descending=True, only_ascii=True)
            down_top = top_tokens(down_proj_vocab, tok, TOP_VOCAB, descending=True, only_ascii=True)
            down_supp = top_tokens(down_proj_vocab, tok, TOP_VOCAB, descending=False, only_ascii=True)

            # "concentration ratio" = top1 |proj| / total |proj| sum
            # 衡量 delta_down 是不是聚焦到单 token (类似 phase25 手工)
            abs_down = down_proj_vocab.abs()
            top1_abs = abs_down.max().item()
            sum_abs = abs_down.sum().item()
            concentration = top1_abs / (sum_abs + 1e-9) * 1000   # 放大 1000 倍便于读

            rows.append({
                "L": L, "I": layer_pool[i]["index"],
                "id": f"L{L}#{layer_pool[i]['index']}",
                "gate_norm": gate_norm, "up_norm": up_norm, "down_norm": down_norm,
                "concentration_x1000": concentration,
                "gate_top_push": gate_top[:5],
                "down_top_push": down_top[:5],
                "down_top_suppress": down_supp[:3],
                "orig_combined_max": layer_pool[i]["combined_max"],
            })

    # sort by down_norm descending (最卖力)
    rows_by_down = sorted(rows, key=lambda r: -r["down_norm"])

    print(f"\n[stats] norm distribution:")
    gate_norms = [r["gate_norm"] for r in rows]
    up_norms = [r["up_norm"] for r in rows]
    down_norms = [r["down_norm"] for r in rows]
    concs = [r["concentration_x1000"] for r in rows]
    def stats(name, vals):
        s = sorted(vals)
        n = len(s)
        return f"  {name}: min={s[0]:.4f}  median={s[n//2]:.4f}  max={s[-1]:.4f}  mean={sum(s)/n:.4f}"
    print(stats("||gate||", gate_norms))
    print(stats("||up||  ", up_norms))
    print(stats("||down||", down_norms))
    print(stats("concentration x1000", concs))

    # ============= 最卖力的 20 个 =============
    out_lines = ["="*80, "  Phase 30: 300 neurons delta 内部分析", "="*80]
    out_lines.append(f"\n层分布: {[(L, len(layers_with_deltas[L])) for L in sorted(layers_with_deltas.keys())]}")
    out_lines.append(f"\n[norms]")
    out_lines.append(stats("||gate||", gate_norms))
    out_lines.append(stats("||up||  ", up_norms))
    out_lines.append(stats("||down||", down_norms))
    out_lines.append(stats("concentration x1000 (越大越聚焦单 token)", concs))

    out_lines.append(f"\n{'='*80}")
    out_lines.append(f"  Top 30 最卖力的神经元 (按 ||delta_down|| 排序)")
    out_lines.append(f"{'='*80}")
    for rank, r in enumerate(rows_by_down[:30], 1):
        gp = [t["tok"] for t in r["gate_top_push"]]
        dp = [t["tok"] for t in r["down_top_push"]]
        ds = [t["tok"] for t in r["down_top_suppress"]]
        out_lines.append(f"\n  #{rank:>2}  {r['id']:<14}  ||gate||={r['gate_norm']:.3f}  "
                         f"||down||={r['down_norm']:.3f}  conc={r['concentration_x1000']:.1f}")
        out_lines.append(f"      gate top push: {gp}")
        out_lines.append(f"      down top push: {dp}")
        out_lines.append(f"      down top supp: {ds}")
        print(f"  #{rank:>2}  {r['id']:<14}  ||down||={r['down_norm']:.3f}  conc={r['concentration_x1000']:.1f}")
        print(f"        gate↑: {gp}")
        print(f"        down↑: {dp}")

    # ============= 按 down concentration 排序找最 "phase25 风格" =============
    rows_by_conc = sorted(rows, key=lambda r: -r["concentration_x1000"])
    out_lines.append(f"\n{'='*80}")
    out_lines.append(f"  Top 20 最聚焦 phase25 风格 (concentration_x1000 最大)")
    out_lines.append(f"{'='*80}")
    for rank, r in enumerate(rows_by_conc[:20], 1):
        dp = [t["tok"] for t in r["down_top_push"]]
        out_lines.append(f"\n  #{rank:>2}  {r['id']:<14}  conc={r['concentration_x1000']:.1f}  "
                         f"||down||={r['down_norm']:.3f}")
        out_lines.append(f"      down top push: {dp}")

    # ============= 沉默的神经元 (||down|| 极小, 训练后基本未改变) =============
    rows_by_down_asc = sorted(rows, key=lambda r: r["down_norm"])
    out_lines.append(f"\n{'='*80}")
    out_lines.append(f"  Bottom 10 最沉默的神经元 (||delta_down|| 极小)")
    out_lines.append(f"{'='*80}")
    for rank, r in enumerate(rows_by_down_asc[:10], 1):
        dp = [t["tok"] for t in r["down_top_push"]]
        out_lines.append(f"  #{rank:>2}  {r['id']:<14}  ||down||={r['down_norm']:.4f}  push: {dp[:3]}")

    # ============= 关键 token 集合统计 =============
    # 看哪些 token 是被多个神经元同时推的 (重点修复 token)
    out_lines.append(f"\n{'='*80}")
    out_lines.append(f"  Token push 频次统计 (前 30 push 频次最高)")
    out_lines.append(f"{'='*80}")
    token_push_count = defaultdict(int)
    for r in rows:
        for t in r["down_top_push"][:3]:   # 每神经元 top-3 push
            token_push_count[t["tok"]] += 1
    sorted_tokens = sorted(token_push_count.items(), key=lambda x: -x[1])
    for tok_str, cnt in sorted_tokens[:30]:
        out_lines.append(f"  {cnt:>3}×  {tok_str!r}")
    print(f"\n[top push tokens across all 300 neurons]:")
    for tok_str, cnt in sorted_tokens[:15]:
        print(f"  {cnt:>3}×  {tok_str!r}")

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "rows": rows[:50],   # 只存前 50 个详情, 全表通过 .txt
        "stats": {
            "gate_norms_summary": stats("||gate||", gate_norms),
            "up_norms_summary": stats("||up||", up_norms),
            "down_norms_summary": stats("||down||", down_norms),
            "concentration_summary": stats("conc", concs),
            "n_neurons": len(rows),
        },
        "top_push_tokens": dict(sorted_tokens[:50]),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
