"""
phase_typology_verify_v3.py — v3 规则在 1030 上的混淆矩阵

v3 规则:
  knowledge:   focus_down >= 0.25 AND align >= 0.80
  functional:  focus_gate >= 0.20 AND focus_down  < 0.25 AND align  < 0.80
  dead_noise:  focus_gate  < 0.17 AND focus_down  < 0.17
  mixed/super: 其他
"""
import sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b\annotated_indicators.db"

CLASSIFY_V3 = """
CASE
    WHEN focus_down >= 0.25 AND align >= 0.80 THEN 'knowledge'
    WHEN focus_gate >= 0.20 AND focus_down  < 0.25 AND align  < 0.80 THEN 'functional'
    WHEN focus_gate  < 0.17 AND focus_down  < 0.17 THEN 'dead_noise'
    ELSE 'mixed'
END
"""

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def show(title, sql, widths=None):
    print(f"\n{title}\n" + "-"*90)
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if widths is None: widths = [max(12, len(c)) for c in cols]
    print("  " + "  ".join(f"{c:>{w}s}" for c, w in zip(cols, widths)))
    for r in rows:
        cells = []
        for c, w in zip(r, widths):
            if c is None: cells.append(f"{'--':>{w}s}")
            elif isinstance(c, float): cells.append(f"{c:{w}.3f}")
            elif isinstance(c, int): cells.append(f"{c:{w}d}")
            else: cells.append(f"{str(c)[:w]:>{w}s}")
        print("  " + "  ".join(cells))


# Q1: 总体
show("Q1. v3 总分布 (vs v2)", f"""
    SELECT {CLASSIFY_V3} AS predicted_class, COUNT(*) AS n,
           ROUND(AVG(focus_gate),3) AS fg, ROUND(AVG(focus_down),3) AS fd, ROUND(AVG(align),3) AS al
    FROM neurons GROUP BY predicted_class ORDER BY n DESC;
""")

# Q2: 混淆矩阵 (ground truth × predicted)
print("\nQ2. v3 混淆矩阵 (横:predicted, 纵:ground truth bucket)\n" + "-"*90)
cur.execute(f"""
    SELECT
        CASE
            WHEN source = 'phase10' THEN 'p10_anxiety'
            WHEN coherence_score >= 9 THEN 'p21_coh9-10'
            WHEN coherence_score >= 7 THEN 'p21_coh7-8'
            WHEN coherence_score >= 4 THEN 'p21_coh4-6'
            WHEN coherence_score >= 2 THEN 'p21_coh2-3'
            ELSE 'p21_coh0-1'
        END AS gt_bucket,
        {CLASSIFY_V3} AS predicted, COUNT(*) AS n
    FROM neurons GROUP BY gt_bucket, predicted;
""")
classes = ["knowledge", "functional", "mixed", "dead_noise"]
buckets = ["p10_anxiety", "p21_coh9-10", "p21_coh7-8", "p21_coh4-6", "p21_coh2-3", "p21_coh0-1"]
mat = {b: {c: 0 for c in classes} for b in buckets}
for gt, pr, n in cur.fetchall():
    if gt in mat and pr in mat[gt]:
        mat[gt][pr] = n
print("  " + f"{'gt_bucket':>12s}  " + "  ".join(f"{c:>11s}" for c in classes) + "  " + f"{'total':>6s}")
for b in buckets:
    cells = [mat[b][c] for c in classes]
    print(f"  {b:>12s}  " + "  ".join(f"{n:11d}" for n in cells) + f"  {sum(cells):6d}")


# Q3: v3 KNOWLEDGE 完整清单
show("Q3. v3 KNOWLEDGE 清单", f"""
    SELECT 'L'||layer||'#'||neuron AS id, source, coherence_score AS coh,
           ROUND(focus_gate,3) AS fg, ROUND(focus_down,3) AS fd, ROUND(align,3) AS al,
           SUBSTR(gate_top10,1,30) AS gate_top, SUBSTR(down_top10,1,30) AS down_top,
           SUBSTR(functional_category,1,15) AS gt_cat
    FROM neurons
    WHERE {CLASSIFY_V3} = 'knowledge'
    ORDER BY align DESC, focus_down DESC;
""", widths=[10, 8, 4, 6, 6, 6, 30, 30, 15])


# Q4: phase10 在 v3 下的命中
show("Q4. phase10 (30 个) v3 分类 — 命中率改善?", f"""
    SELECT {CLASSIFY_V3} AS predicted, COUNT(*) AS n,
           ROUND(100.0*COUNT(*)/30.0, 1) AS pct
    FROM neurons WHERE source='phase10' GROUP BY predicted ORDER BY n DESC;
""", widths=[14, 4, 6])


# Q5: phase10 详细 — 哪些被 v3 正确识别
show("Q5. phase10 详细映射 (gt_category → v3_predicted)", f"""
    SELECT functional_category AS gt_cat, {CLASSIFY_V3} AS predicted, COUNT(*) AS n
    FROM neurons WHERE source='phase10'
    GROUP BY gt_cat, predicted ORDER BY gt_cat, n DESC;
""", widths=[28, 14, 4])


# Q6: 极端样本 — v3 新发现的 knowledge (相比 v2)
show("Q6. v3 新增 knowledge 样本 (v2 下是 mixed/other)", f"""
    SELECT 'L'||layer||'#'||neuron AS id, source, coherence_score AS coh,
           ROUND(focus_gate,3) AS fg, ROUND(focus_down,3) AS fd, ROUND(align,3) AS al,
           gate_top10
    FROM neurons
    WHERE {CLASSIFY_V3} = 'knowledge'
      AND NOT (focus_gate >= 0.25 AND focus_down >= 0.30 AND align >= 0.80);
""", widths=[10, 8, 4, 6, 6, 6, 40])


# Q7: align 高但 focus_down 低的"假 knowledge"被排除了吗 (sanity)
show("Q7. align>=0.85 但 focus_down<0.25 的样本 (v3 应不归 knowledge)", f"""
    SELECT 'L'||layer||'#'||neuron AS id, source, coherence_score AS coh,
           ROUND(focus_gate,3) AS fg, ROUND(focus_down,3) AS fd, ROUND(align,3) AS al,
           {CLASSIFY_V3} AS predicted
    FROM neurons
    WHERE align >= 0.85 AND focus_down < 0.25
    ORDER BY align DESC LIMIT 10;
""", widths=[10, 8, 4, 6, 6, 6, 12])


# Q8: 灰色地带 — mixed 数量 + 看典型
show("Q8. v3 'mixed' 桶里的典型样本 (前 15)", f"""
    SELECT 'L'||layer||'#'||neuron AS id, source, coherence_score AS coh,
           ROUND(focus_gate,3) AS fg, ROUND(focus_down,3) AS fd, ROUND(align,3) AS al,
           SUBSTR(functional_category||COALESCE(gate_theme,''),1,20) AS theme
    FROM neurons
    WHERE {CLASSIFY_V3} = 'mixed'
    ORDER BY focus_down DESC LIMIT 15;
""", widths=[10, 8, 4, 6, 6, 6, 22])

conn.close()
print("\n[Done]")
