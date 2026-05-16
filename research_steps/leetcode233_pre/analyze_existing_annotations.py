"""
analyze_existing_annotations.py — 先读透现有 phase10/21 标注,不算指标
仅汇总 ground truth, 看分布特征
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8")
from collections import Counter, defaultdict

P10 = r"J:\amr\amr_wtf\outputs\gemma_code_GB01\phase10_annotations.jsonl"
P21 = r"J:\amr\amr_wtf\outputs\gemma_code_GB01\phase21_annotations_full.json"

# 1. Phase 10: 30 个 K=30 焦虑/功能神经元
print("="*70)
print("Phase 10 (K=30 RLHF anxiety neurons, found via clamp-diff)")
print("="*70)
p10 = []
with open(P10, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            p10.append(json.loads(line))
print(f"Total: {len(p10)} neurons")
print()
print("Functional categories:")
cats = Counter(n["annotation"]["functional_category"] for n in p10)
for c, ct in cats.most_common():
    print(f"  {c:35s}  {ct}")
print()
print("Anxiety scores distribution:")
scores = Counter(n["annotation"]["rlhf_anxiety_score"] for n in p10)
for s in sorted(scores):
    print(f"  score={s}  {scores[s]} neurons")
print()
print("All 30 neurons:")
print(f"  {'id':12s}  {'category':30s}  {'anxiety':3s}  trigger_condition")
print(f"  {'-'*12}  {'-'*30}  {'-'*3}  {'-'*40}")
for n in p10:
    a = n["annotation"]
    trig = a.get("trigger_condition", "")[:60]
    print(f"  {n['id']:12s}  {a['functional_category']:30s}  {a['rlhf_anxiety_score']:3d}  {trig}")
print()

# Gate vs Down theme consistency (key question for knowledge vs functional)
print("Gate vs Down themes (looking for read=write match vs mismatch):")
for n in p10[:10]:
    a = n["annotation"]
    gate_top = ", ".join(n.get("raw_gate_top_push", [])[:5])
    down_push = ", ".join(n.get("raw_down_top_push", [])[:5])
    down_supp = ", ".join(n.get("raw_down_top_suppress", [])[:5])
    print(f"  {n['id']}: ")
    print(f"    gate top push : {gate_top}")
    print(f"    down top push : {down_push}")
    print(f"    down top suppr: {down_supp}")
    print()

# 2. Phase 21: 1000 个 inert 神经元标注
print("="*70)
print("Phase 21 (1000 inert candidates)")
print("="*70)
with open(P21, "r", encoding="utf-8") as f:
    p21_meta = json.load(f)
print(f"Annotation count: {p21_meta.get('annotation_count', 'N/A')}")

# 从 chunks 收集所有 annotations
chunks_dir = r"J:\amr\amr_wtf\outputs\gemma_code_GB01\phase21_chunks"
p21 = []
for fname in sorted(os.listdir(chunks_dir)):
    if not fname.endswith(".json"): continue
    with open(os.path.join(chunks_dir, fname), "r", encoding="utf-8") as f:
        d = json.load(f)
        p21.extend(d.get("annotations", []))
print(f"Loaded from chunks: {len(p21)}")
print()

# Coherence distribution
print("Coherence score distribution (0=incoherent, 10=clear theme):")
coh = Counter(n.get("coherence_score") for n in p21)
total_p21 = sum(coh.values())
for s in sorted(coh.keys(), key=lambda x: (x is None, x)):
    if s is None: continue
    pct = coh[s] / total_p21 * 100
    bar = "#" * int(pct)
    print(f"  {s:2}  {coh[s]:4} ({pct:4.1f}%)  {bar}")
print()

# safe_to_overwrite distribution
safe = Counter(n.get("safe_to_overwrite") for n in p21)
print("safe_to_overwrite:")
for k, v in safe.items():
    print(f"  {k}: {v}")
print()

# Repurpose risk
risk = Counter(n.get("repurpose_risk") for n in p21)
print("repurpose_risk:")
for k, v in risk.items():
    print(f"  {k}: {v}")
print()

# Top gate/down themes (text)
print("Top gate_theme (most common):")
gt = Counter(n.get("gate_theme", "") for n in p21)
for t, c in gt.most_common(15):
    if t:
        print(f"  {c:4}  {t[:70]}")
print()
print("Top down_theme (most common):")
dt = Counter(n.get("down_theme", "") for n in p21)
for t, c in dt.most_common(15):
    if t:
        print(f"  {c:4}  {t[:70]}")
print()

# 关键统计: coherence vs gate-down theme 关系
print("Key insight: for high coherence (>=7), are gate_theme == down_theme?")
for thresh in [7, 8, 9, 10]:
    hi_coh = [n for n in p21 if (n.get("coherence_score") or 0) >= thresh]
    same_theme = sum(1 for n in hi_coh if n.get("gate_theme") == n.get("down_theme"))
    incoh_gate_or_down = sum(1 for n in hi_coh
                             if "incoh" in (n.get("gate_theme") or "").lower()
                             or "incoh" in (n.get("down_theme") or "").lower())
    print(f"  coherence >= {thresh}: {len(hi_coh)} neurons, "
          f"gate==down: {same_theme}, gate or down incoherent: {incoh_gate_or_down}")
print()

print("Some high-coherence neurons (probable knowledge):")
hi_coh = sorted([n for n in p21 if (n.get("coherence_score") or 0) >= 8],
                key=lambda x: -x.get("coherence_score", 0))[:15]
for n in hi_coh:
    print(f"  {n['id']:12s}  coh={n.get('coherence_score'):2d}  "
          f"gate='{(n.get('gate_theme') or '')[:30]:30s}'  "
          f"down='{(n.get('down_theme') or '')[:30]:30s}'")
print()

print("Some low-coherence neurons (probable dead/noise):")
lo_coh = [n for n in p21 if (n.get("coherence_score") or 0) <= 1][:15]
for n in lo_coh:
    print(f"  {n['id']:12s}  coh={n.get('coherence_score'):2d}  "
          f"gate='{(n.get('gate_theme') or '')[:30]:30s}'  "
          f"down='{(n.get('down_theme') or '')[:30]:30s}'")
print()

# 3. 层分布
print("="*70)
print("Layer distribution of annotated neurons:")
print("="*70)
p10_layers = Counter(n["layer"] for n in p10)
p21_layers = Counter(int(n["id"].split("L")[1].split("#")[0]) for n in p21)
print(f"  {'layer':6s}  {'p10 (anxiety)':14s}  {'p21 (inert)':12s}")
for L in sorted(set(list(p10_layers.keys()) + list(p21_layers.keys()))):
    print(f"  L{L:02d}     {p10_layers.get(L,0):14d}  {p21_layers.get(L,0):12d}")
