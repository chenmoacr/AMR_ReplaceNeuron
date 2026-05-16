"""
phase_typology_verify.py — Step A: 在 1030 个标注神经元上验证 v2 判别规则

判别规则 (基于 v2 数据校准):
  knowledge:   focus_gate >= 0.25 AND focus_down >= 0.30 AND align >= 0.80
  functional:  focus_gate >= 0.20 AND focus_down  < 0.20 AND align  < 0.70
  dead_noise:  focus_gate  < 0.18 AND focus_down  < 0.18
  mixed:       其他 (包括 superposition 嫌疑)

验证目标:
  Q1. 总体分布
  Q2. phase10 (30 个已知 functional) 我们识别出多少?
  Q3. phase21 各 coherence 桶在我们规则下分布
  Q4. 找出 predicted_knowledge 清单 (重要发现)
  Q5. 混淆点: ground truth 说 X 但我们说 Y 的样本
"""
import os, sys, sqlite3, json
sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b\annotated_indicators.db"
OUT_MD  = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b\verify_step_a.md"

# 判别规则 SQL CASE
CLASSIFY_SQL = """
CASE
    WHEN focus_gate >= 0.25 AND focus_down >= 0.30 AND align >= 0.80 THEN 'knowledge'
    WHEN focus_gate >= 0.20 AND focus_down  < 0.20 AND align  < 0.70 THEN 'functional'
    WHEN focus_gate  < 0.18 AND focus_down  < 0.18                   THEN 'dead_noise'
    ELSE 'mixed'
END
"""

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def query(sql, params=()):
    cur.execute(sql, params)
    return cur.description, cur.fetchall()

def print_table(title, desc, rows, widths=None):
    print(f"\n{title}")
    print("-" * 90)
    cols = [d[0] for d in desc]
    if widths is None:
        widths = [max(15, len(c)) for c in cols]
    print("  " + "  ".join(f"{c:>{w}s}" for c, w in zip(cols, widths)))
    for r in rows:
        cells = []
        for c, w in zip(r, widths):
            if c is None:
                cells.append(f"{'--':>{w}s}")
            elif isinstance(c, float):
                cells.append(f"{c:{w}.4f}")
            elif isinstance(c, int):
                cells.append(f"{c:{w}d}")
            else:
                s = str(c)[:w]
                cells.append(f"{s:>{w}s}")
        print("  " + "  ".join(cells))


# ──────────────────────────────────────────────────────────────────────
# Q1: 总体分类分布
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT
        {CLASSIFY_SQL} AS predicted_class,
        COUNT(*) AS n,
        ROUND(AVG(focus_gate), 3) AS fg,
        ROUND(AVG(focus_down), 3) AS fd,
        ROUND(AVG(align), 3) AS al
    FROM neurons
    GROUP BY predicted_class
    ORDER BY n DESC;
""")
print_table("Q1. 总体分类分布 (n=1030)", desc, rows)


# ──────────────────────────────────────────────────────────────────────
# Q2: phase10 (30 个已知 functional) 命中情况
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT
        functional_category,
        {CLASSIFY_SQL} AS predicted_class,
        COUNT(*) AS n
    FROM neurons
    WHERE source = 'phase10'
    GROUP BY functional_category, predicted_class
    ORDER BY functional_category, n DESC;
""")
print_table("Q2. phase10 (30 个 ground-truth functional) 按 category vs 预测",
            desc, rows, widths=[25, 18, 5])


# ──────────────────────────────────────────────────────────────────────
# Q3: phase21 coherence 桶在我们规则下的分类
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT
        CASE
            WHEN coherence_score >= 9 THEN 'coh_9-10'
            WHEN coherence_score >= 7 THEN 'coh_7-8'
            WHEN coherence_score >= 4 THEN 'coh_4-6'
            WHEN coherence_score >= 2 THEN 'coh_2-3'
            ELSE 'coh_0-1'
        END AS coh_bucket,
        {CLASSIFY_SQL} AS predicted_class,
        COUNT(*) AS n
    FROM neurons
    WHERE source = 'phase21'
    GROUP BY coh_bucket, predicted_class
    ORDER BY coh_bucket DESC, n DESC;
""")
print_table("Q3. phase21 coherence 桶 vs 预测分类", desc, rows, widths=[12, 18, 5])


# ──────────────────────────────────────────────────────────────────────
# Q4: predicted_knowledge 全部清单 — 这是核心研究产出
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT
        'L' || layer || '#' || neuron AS id,
        source,
        coherence_score AS coh,
        ROUND(focus_gate, 3) AS fg,
        ROUND(focus_down, 3) AS fd,
        ROUND(align, 3) AS al,
        SUBSTR(gate_top10, 1, 35) AS gate_top,
        SUBSTR(down_top10, 1, 35) AS down_top
    FROM neurons
    WHERE {CLASSIFY_SQL} = 'knowledge'
    ORDER BY align DESC, focus_down DESC;
""")
print_table("Q4. predicted KNOWLEDGE 神经元清单 (★ 核心发现 ★)",
            desc, rows, widths=[10, 8, 4, 6, 6, 6, 35, 35])


# ──────────────────────────────────────────────────────────────────────
# Q5: 已知 functional (phase10) 我们漏识别的 (predicted != functional)
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT
        'L' || layer || '#' || neuron AS id,
        functional_category AS gt_cat,
        anxiety_score AS anx,
        {CLASSIFY_SQL} AS predicted,
        ROUND(focus_gate, 3) AS fg,
        ROUND(focus_down, 3) AS fd,
        ROUND(align, 3) AS al,
        SUBSTR(gate_top10, 1, 25) AS gate_top
    FROM neurons
    WHERE source = 'phase10' AND {CLASSIFY_SQL} != 'functional'
    ORDER BY predicted, focus_gate DESC;
""")
print_table("Q5. phase10 漏判: ground-truth functional 但我们预测为别的",
            desc, rows, widths=[10, 20, 4, 12, 6, 6, 6, 25])


# ──────────────────────────────────────────────────────────────────────
# Q6: 完美命中: phase10 = functional 且我们预测 = functional
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT COUNT(*) AS hits, ROUND(100.0*COUNT(*)/30.0, 1) AS pct
    FROM neurons
    WHERE source = 'phase10' AND {CLASSIFY_SQL} = 'functional';
""")
print_table("Q6. phase10 命中 functional 的数量 (30 个中)", desc, rows, widths=[6, 6])


# ──────────────────────────────────────────────────────────────────────
# Q7: 高 coherence (9-10) 在我们规则下分到了哪
# ──────────────────────────────────────────────────────────────────────
desc, rows = query(f"""
    SELECT
        'L' || layer || '#' || neuron AS id,
        coherence_score AS coh,
        {CLASSIFY_SQL} AS predicted,
        ROUND(focus_gate, 3) AS fg,
        ROUND(focus_down, 3) AS fd,
        ROUND(align, 3) AS al,
        SUBSTR(gate_theme, 1, 25) AS gate_theme,
        SUBSTR(down_theme, 1, 25) AS down_theme
    FROM neurons
    WHERE source = 'phase21' AND coherence_score >= 9 AND {CLASSIFY_SQL} != 'knowledge'
    ORDER BY align DESC
    LIMIT 15;
""")
print_table("Q7. phase21 coherence>=9 但未被识别为 knowledge 的样本",
            desc, rows, widths=[10, 4, 12, 6, 6, 6, 25, 25])


# ──────────────────────────────────────────────────────────────────────
# Q8: focus_down 阈值敏感性: 不同阈值下 knowledge 数量
# ──────────────────────────────────────────────────────────────────────
print("\nQ8. 阈值敏感性扫描 (调整 focus_down 和 align)")
print("-" * 90)
print("  " + "  ".join(f"{c:>15s}" for c in
                       ["fd_thresh", "align_thresh", "n_knowledge", "phase21_pct", "phase10_pct"]))
for fd_th in [0.20, 0.25, 0.30, 0.35]:
    for al_th in [0.70, 0.75, 0.80, 0.85]:
        cur.execute(f"""
            SELECT
                COUNT(*) total,
                SUM(CASE WHEN source='phase21' THEN 1 ELSE 0 END) p21,
                SUM(CASE WHEN source='phase10' THEN 1 ELSE 0 END) p10
            FROM neurons
            WHERE focus_gate >= 0.20 AND focus_down >= ? AND align >= ?
        """, (fd_th, al_th))
        total, p21, p10 = cur.fetchone()
        print(f"  {fd_th:15.2f}  {al_th:15.2f}  {total:15d}  "
              f"{(p21/1000*100):14.1f}%  {(p10/30*100):14.1f}%")


# ──────────────────────────────────────────────────────────────────────
# Q9: Confusion-matrix 风格汇总
# ──────────────────────────────────────────────────────────────────────
print("\nQ9. 总体混淆矩阵 (横: predicted_class, 纵: ground truth bucket)")
print("-" * 90)
desc, rows = query(f"""
    SELECT
        CASE
            WHEN source = 'phase10' THEN 'p10_anxiety'
            WHEN coherence_score >= 9 THEN 'p21_coh9-10'
            WHEN coherence_score >= 7 THEN 'p21_coh7-8'
            WHEN coherence_score >= 4 THEN 'p21_coh4-6'
            WHEN coherence_score >= 2 THEN 'p21_coh2-3'
            ELSE 'p21_coh0-1'
        END AS gt_bucket,
        {CLASSIFY_SQL} AS predicted,
        COUNT(*) AS n
    FROM neurons
    GROUP BY gt_bucket, predicted
    ORDER BY gt_bucket DESC, n DESC;
""")
# Pivot to wide format
classes = ["knowledge", "functional", "mixed", "dead_noise"]
buckets = ["p10_anxiety", "p21_coh9-10", "p21_coh7-8", "p21_coh4-6", "p21_coh2-3", "p21_coh0-1"]
matrix = {b: {c: 0 for c in classes} for b in buckets}
for gt, pred, n in rows:
    if gt in matrix and pred in matrix[gt]:
        matrix[gt][pred] = n
print("  " + f"{'gt_bucket':>12s}  " + "  ".join(f"{c:>11s}" for c in classes) + "  " + f"{'total':>6s}")
for b in buckets:
    cells = [matrix[b][c] for c in classes]
    total = sum(cells)
    print(f"  {b:>12s}  " + "  ".join(f"{n:11d}" for n in cells) + f"  {total:6d}")

conn.close()
print("\n[Done]", flush=True)
