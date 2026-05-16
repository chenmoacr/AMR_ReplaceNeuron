"""
Phase 20b: 用相对阈值 (cot/resp ratio) 重测 'cot 神经元 vs response 神经元 分开' 假设.

Phase 20 用绝对阈值 0.5/0.05 → 所有 cot-only/resp-only 都是 0.
但 Gemma 4 在 K=30 clamp 下噪声底就 ~0.08, 绝对死几乎不存在.

相对测试:
  cot-dominant  = cot_max > 0.5 AND resp_max / cot_max < 0.1
  resp-dominant = resp_max > 0.5 AND cot_max / resp_max < 0.1
  balanced      = both > 0.5 AND ratio in [0.5, 2.0]
  both quiet    = both < 0.2 (近似 'dead enough')
"""
import os, sys, json
from pathlib import Path
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch

ROOT = Path("J:/amr/amr_wtf")
SCAN_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase20_dead_scan.pt"
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase20b_relative.txt"


def main():
    print(f"[load] {SCAN_PT}")
    payload = torch.load(SCAN_PT, weights_only=False)
    per_layer = payload["per_layer"]
    layer_dims = payload["layer_dims"]
    clamped_set = set(tuple(x) for x in payload["clamped_set"])

    n_layers = len(per_layer)
    print(f"  {n_layers} layers, total neurons = {sum(layer_dims.values())}")

    # Build flat (L, I, cot_max, resp_max)
    records = []
    for L in range(n_layers):
        cm = per_layer[L]["cot_max"]
        rm = per_layer[L]["resp_max"]
        for I in range(cm.shape[0]):
            records.append((L, I, float(cm[I].item()), float(rm[I].item())))
    total = len(records)
    print(f"  built {total} records")

    # ============= Relative specialization (4 quadrants by ratio) =============
    out_lines = []
    out_lines.append("="*80)
    out_lines.append("  Phase 20b: 相对阈值 cot/response 专属性分析")
    out_lines.append("="*80)

    # ratio test: 用 epsilon 防 0
    eps = 1e-4
    def ratio_class(c, r):
        if c < 0.2 and r < 0.2:
            return "both_quiet"
        if c > 0.5 and r / (c + eps) < 0.1:
            return "cot_dominant"
        if r > 0.5 and c / (r + eps) < 0.1:
            return "resp_dominant"
        if c > 0.5 and r > 0.5:
            ratio = max(c, r) / min(c, r)
            if ratio < 2.0:
                return "balanced_alive"
            elif ratio < 10.0:
                return "lean_alive"
            else:
                # one side significantly stronger but other not "dead enough"
                return "skewed_alive"
        return "mixed_zone"   # one side alive one side quiet but not extreme

    counts = Counter()
    classified = []
    for L, I, c, r in records:
        cls = ratio_class(c, r)
        counts[cls] += 1
        classified.append((L, I, c, r, cls))

    out_lines.append(f"\n相对 4 象限 (ratio-based):")
    out_lines.append(f"  cot_dominant   (cot>0.5 & resp/cot<0.1) : {counts['cot_dominant']:>7}  "
                     f"({100*counts['cot_dominant']/total:.2f}%)  ← cot 专属")
    out_lines.append(f"  resp_dominant  (resp>0.5 & cot/resp<0.1): {counts['resp_dominant']:>7}  "
                     f"({100*counts['resp_dominant']/total:.2f}%)  ← response 专属")
    out_lines.append(f"  balanced_alive (both>0.5 & ratio<2x)    : {counts['balanced_alive']:>7}  "
                     f"({100*counts['balanced_alive']/total:.2f}%)")
    out_lines.append(f"  lean_alive     (both>0.5 & 2x<ratio<10x): {counts['lean_alive']:>7}  "
                     f"({100*counts['lean_alive']/total:.2f}%)")
    out_lines.append(f"  skewed_alive   (both>0.5 & ratio>10x)   : {counts['skewed_alive']:>7}  "
                     f"({100*counts['skewed_alive']/total:.2f}%)")
    out_lines.append(f"  both_quiet     (both<0.2)               : {counts['both_quiet']:>7}  "
                     f"({100*counts['both_quiet']/total:.2f}%)  ← 主候选区")
    out_lines.append(f"  mixed_zone     (one alive one quiet/borderline): {counts['mixed_zone']:>7}  "
                     f"({100*counts['mixed_zone']/total:.2f}%)")

    verdict = "SUPPORTED" if (counts['cot_dominant'] + counts['resp_dominant']) > total * 0.005 else "WEAK"
    out_lines.append(f"\n→ 用户 'cot/response 分开' 假设: {verdict}")
    out_lines.append(f"  专属神经元 (cot_dom + resp_dom) = "
                     f"{counts['cot_dominant'] + counts['resp_dominant']} "
                     f"({100*(counts['cot_dominant'] + counts['resp_dominant'])/total:.2f}%)")

    # ============= cot_dominant 例子 =============
    cot_doms = [r for r in classified if r[4] == "cot_dominant"]
    cot_doms.sort(key=lambda r: -(r[2]))  # 按 cot_max 降序
    out_lines.append(f"\n--- cot_dominant top 20 (cot 强、response 弱) ---")
    out_lines.append(f"  {'rank':>4}  {'neuron':<14} {'cot_max':>10} {'resp_max':>10} {'ratio':>8}")
    for r in cot_doms[:20]:
        L, I, c, rsp, _ = r
        ratio = c / max(rsp, eps)
        out_lines.append(f"  L{L:02d}#{I:<8d}  {c:>10.3f}  {rsp:>10.3f}  {ratio:>8.1f}x")
    # 按层分布
    L_dist_cot = Counter(r[0] for r in cot_doms)
    out_lines.append(f"\n  cot_dominant 按层分布:")
    for L in sorted(L_dist_cot.keys()):
        out_lines.append(f"    L{L:02d}: {L_dist_cot[L]}")

    # ============= resp_dominant 例子 =============
    resp_doms = [r for r in classified if r[4] == "resp_dominant"]
    resp_doms.sort(key=lambda r: -(r[3]))
    out_lines.append(f"\n--- resp_dominant top 20 (response 强、cot 弱) ---")
    out_lines.append(f"  {'rank':>4}  {'neuron':<14} {'cot_max':>10} {'resp_max':>10} {'ratio':>8}")
    for r in resp_doms[:20]:
        L, I, c, rsp, _ = r
        ratio = rsp / max(c, eps)
        out_lines.append(f"  L{L:02d}#{I:<8d}  {c:>10.3f}  {rsp:>10.3f}  {ratio:>8.1f}x")
    L_dist_resp = Counter(r[0] for r in resp_doms)
    out_lines.append(f"\n  resp_dominant 按层分布:")
    for L in sorted(L_dist_resp.keys()):
        out_lines.append(f"    L{L:02d}: {L_dist_resp[L]}")

    # ============= both_quiet (Phase 22 候选源) =============
    both_quiet = [r for r in classified if r[4] == "both_quiet"]
    out_lines.append(f"\n--- both_quiet 池 (combined<0.4 for full pool, top 10 coldest) ---")
    both_quiet.sort(key=lambda r: r[2] + r[3])
    for r in both_quiet[:10]:
        L, I, c, rsp, _ = r
        out_lines.append(f"  L{L:02d}#{I:<8d}  cot={c:.4f}  resp={rsp:.4f}  combined={c+rsp:.4f}")
    L_dist_quiet = Counter(r[0] for r in both_quiet)
    out_lines.append(f"\n  both_quiet 按层分布:")
    for L in sorted(L_dist_quiet.keys()):
        n = L_dist_quiet[L]
        bar = "█" * (n // 100)
        out_lines.append(f"    L{L:02d}: {n:>6} {bar}")
    out_lines.append(f"\n  both_quiet 总数 = {len(both_quiet)}")

    # ============= verdict & next-step recommendation =============
    out_lines.append("\n" + "="*80)
    out_lines.append("  结论 & next step")
    out_lines.append("="*80)
    out_lines.append(f"1. 用户 'cot/response 神经元分开' 假设: {verdict}")
    out_lines.append(f"   - cot_dominant: {counts['cot_dominant']} 个 (cot 比 response 强 10x+)")
    out_lines.append(f"   - resp_dominant: {counts['resp_dominant']} 个 (response 比 cot 强 10x+)")
    out_lines.append(f"   - 严格绝对死 (<0.05) 几乎不存在 (3/337920)")
    out_lines.append(f"   - 但 'both_quiet (<0.2)' 池 = {len(both_quiet)} 个, "
                     f"足够支撑 Phase 21 后续标注")
    out_lines.append(f"")
    out_lines.append(f"2. Phase 21 候选源选择:")
    out_lines.append(f"   - 主选: both_quiet 池 (~{len(both_quiet)}), 改 W_gate/W_down 不影响 cot/resp 任一路径")
    out_lines.append(f"   - 次选: cot_dominant (~{counts['cot_dominant']}), 但改了会扰乱原本 cot 功能 → 风险高")
    out_lines.append(f"   - Phase 20 已选的 1000 个 (combined<0.2) 大多 ⊂ both_quiet, 可直接用")

    text = "\n".join(out_lines)
    print(text)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
