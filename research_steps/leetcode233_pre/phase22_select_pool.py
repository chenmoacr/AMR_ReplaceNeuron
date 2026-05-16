"""
Phase 22: 从 1000 标注里选 200 inert pool.

筛选条件 (依次):
  1. safe_to_overwrite = True
  2. coherence_score <= 3 (strict)
  3. repurpose_risk in ('low',)
  4. ||W_down|| > 0.5  (避免推不动的太小神经元)
  5. layer in [13, 19] (主选), L20+ 也允许做后备
  6. 层均衡 + L16 cap

输出: phase22_inert_pool_200.json
"""
import os, sys, json, glob
from pathlib import Path
from collections import Counter, defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
CHUNK_DIR = ROOT / "outputs" / "gemma_code_GB01" / "phase21_chunks"
INERT_DATA_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase21_inert_for_gemini.json"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase22_inert_pool_200.json"
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase22_summary.txt"

TARGET_N = 200
L16_CAP = 100        # L16 最多取 100 (不要全选 L16)
MIN_W_DOWN_NORM = 0.5
MAX_COHERENCE = 3
ALLOWED_RISK = {"low"}
LAYER_PRIORITY = list(range(13, 20))   # L13-L19 优先


def main():
    # Load all chunks
    anns_by_id = {}
    for f in sorted(glob.glob(str(CHUNK_DIR / "chunk_*.json"))):
        d = json.loads(open(f, encoding="utf-8").read())
        for ann in d["annotations"]:
            anns_by_id[ann.get("id")] = ann
    print(f"[load] {len(anns_by_id)} annotations")

    # Load original neuron data (with W_gate/W_down norms + projections)
    neurons = json.loads(INERT_DATA_PATH.read_text(encoding="utf-8"))
    n_by_id = {n["id"]: n for n in neurons}
    print(f"[load] {len(neurons)} neuron data records")

    # ============= 过滤 =============
    passed = []
    reject_reason = Counter()
    for n in neurons:
        ann = anns_by_id.get(n["id"])
        if ann is None:
            reject_reason["no_annotation"] += 1
            continue
        if not ann.get("safe_to_overwrite"):
            reject_reason["not_safe"] += 1
            continue
        coh = ann.get("coherence_score", 11)
        if coh > MAX_COHERENCE:
            reject_reason[f"coh>{MAX_COHERENCE}"] += 1
            continue
        if ann.get("repurpose_risk") not in ALLOWED_RISK:
            reject_reason["not_low_risk"] += 1
            continue
        if n["W_down_col_norm"] < MIN_W_DOWN_NORM:
            reject_reason["W_down_too_small"] += 1
            continue
        # passed
        passed.append({
            **n,
            "annotation": ann,
            "ann_coherence": coh,
            "ann_safe": True,
            "ann_risk": ann.get("repurpose_risk"),
            "ann_gate_theme": ann.get("gate_theme"),
            "ann_down_theme": ann.get("down_theme"),
            "ann_notes": ann.get("notes"),
        })
    print(f"[filter] passed {len(passed)}/{len(neurons)} after strict criteria")
    print(f"[reject reasons]")
    for k, v in reject_reason.most_common():
        print(f"  {k}: {v}")

    # 层分布
    by_layer = defaultdict(list)
    for p in passed:
        by_layer[p["layer"]].append(p)
    # 每层按 combined_max 升序 (最冷的优先)
    for L in by_layer:
        by_layer[L].sort(key=lambda p: p["combined_max"])

    print(f"\n[passed by layer]")
    for L in sorted(by_layer.keys()):
        print(f"  L{L:02d}: {len(by_layer[L])}")

    # ============= 选 200, L16 cap, 其他层尽量取满 =============
    selected = []
    remaining = TARGET_N

    # 优先取 L13-L19 (排除 L16) 全部
    for L in LAYER_PRIORITY:
        if L == 16:
            continue
        for p in by_layer.get(L, []):
            if remaining > 0:
                selected.append(p)
                remaining -= 1
            else:
                break
    print(f"\n[select] non-L16 from L13-L19: {len(selected)}, remaining target {remaining}")

    # 然后 L16 (cap)
    n_l16 = min(L16_CAP, remaining, len(by_layer.get(16, [])))
    for p in by_layer.get(16, [])[:n_l16]:
        selected.append(p)
    remaining -= n_l16
    print(f"[select] L16 added: {n_l16}, remaining target {remaining}")

    # 还不够的话, 用 L20+ 或 <L13
    if remaining > 0:
        backup_layers = [L for L in sorted(by_layer.keys()) if L < 13 or L > 19]
        for L in backup_layers:
            for p in by_layer.get(L, []):
                if remaining > 0:
                    selected.append(p)
                    remaining -= 1
                else:
                    break
        print(f"[select] backup layers added, remaining {remaining}")

    # 还不够? 提高 L16 cap
    if remaining > 0:
        more_l16 = [p for p in by_layer.get(16, [])[n_l16:] if p not in selected]
        for p in more_l16:
            if remaining > 0:
                selected.append(p)
                remaining -= 1
            else:
                break
        print(f"[select] more L16 to fill, remaining {remaining}")

    print(f"\n[final] selected {len(selected)} (target {TARGET_N})")

    # 最终统计
    selected_by_layer = Counter(p["layer"] for p in selected)
    print(f"\n[final layer distribution]")
    for L in sorted(selected_by_layer.keys()):
        bar = "█" * (selected_by_layer[L] // 3)
        print(f"  L{L:02d}: {selected_by_layer[L]:>3} {bar}")

    coh_dist = Counter(p["ann_coherence"] for p in selected)
    print(f"\n[selected coherence dist] {dict(coh_dist)}")

    # 排序输出 (按 combined_max 升序)
    selected.sort(key=lambda p: p["combined_max"])

    # 写文件 (精简版, 不带全部 vocab projection 防文件爆)
    out_data = []
    for p in selected:
        out_data.append({
            "rank": p["rank"],
            "id": p["id"], "layer": p["layer"], "index": p["index"],
            "cot_max": p["cot_max"], "resp_max": p["resp_max"],
            "combined_max": p["combined_max"],
            "W_gate_row_norm": p["W_gate_row_norm"],
            "W_down_col_norm": p["W_down_col_norm"],
            "ann_coherence": p["ann_coherence"],
            "ann_safe": p["ann_safe"],
            "ann_risk": p["ann_risk"],
            "ann_gate_theme": p["ann_gate_theme"],
            "ann_down_theme": p["ann_down_theme"],
            "ann_notes": p["ann_notes"],
            "gate_top_push": [x["tok"] for x in p["gate_vocab_top_push"][:5]],
            "down_top_push": [x["tok"] for x in p["down_vocab_top_push"][:5]],
        })

    OUT_PATH.write_text(json.dumps(out_data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}  ({OUT_PATH.stat().st_size//1024} KB)")

    out_lines = [
        "="*80,
        f"  Phase 22: 200 inert pool selection",
        "="*80,
        f"\n选 {len(selected)} / {len(neurons)} 总候选",
        f"通过过滤: {len(passed)}, 拒绝原因: {dict(reject_reason.most_common())}",
        f"\n--- 层分布 ---",
    ]
    for L in sorted(selected_by_layer.keys()):
        out_lines.append(f"  L{L:02d}: {selected_by_layer[L]}")
    out_lines.append(f"\n--- coherence 分布 ---")
    for c in sorted(coh_dist.keys()):
        out_lines.append(f"  coh={c}: {coh_dist[c]}")
    out_lines.append(f"\n--- top 10 coldest selected ---")
    for p in selected[:10]:
        out_lines.append(f"  {p['id']:<14} cot={p['cot_max']:.4f}  resp={p['resp_max']:.4f}  "
                         f"coh={p['ann_coherence']}  "
                         f"gate={p['ann_gate_theme'][:30]}  down={p['ann_down_theme'][:30]}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
