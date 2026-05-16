"""
phase_typology_v2.py — 三维指标 (focus_gate, focus_down, align) 算法 + SQLite 存储

新指标定义:
  focus_gate = mean pairwise cosine of top-K gate-projected token embeddings
               (越高 = gate 读的 token 在同一语义场, 单义)
  focus_down = 同上但 W_down (越高 = down 写的 token 单义)
  align      = cos(center(gate_top_K_emb), center(down_top_K_emb))
               (越高 = 读和写在同一主题方向)

预期分类形态:
  knowledge:   focus_gate 高 ∧ focus_down 高 ∧ align > 0.3
  functional:  focus_gate 高 ∧ (focus_down 低 ∨ align < 0.1)
  dead:        ||W_down|| 很低
  superposition: focus_gate 低 ∧ focus_down 低 ∧ ||W_down|| 不低

目标:
  对 1030 个已标注神经元算指标, 看 ground truth 类别在指标空间中是否真的分开.
"""
import os, sys, json, struct, time, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
import torch
import torch.nn.functional as F

E2B_PATH    = r"J:\amr\models\gemma-4-E2B-it\model.safetensors"
TOK_PATH    = r"J:\amr\models\gemma-4-E2B-it"
P10_PATH    = r"J:\amr\amr_wtf\outputs\gemma_code_GB01\phase10_annotations.jsonl"
P21_CHUNKS  = r"J:\amr\amr_wtf\outputs\gemma_code_GB01\phase21_chunks"
OUT_DIR     = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b"
DB_PATH     = os.path.join(OUT_DIR, "annotated_indicators.db")

K           = 30
DEVICE      = "cuda:0"
DTYPE_MAP   = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Tokenizer (用于 decode top-K token 字符串, 写进 db 方便人工核查) ──
print("[1] Loading tokenizer...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)


# ── 2. Read safetensors header ────────────────────────────────────────
print("\n[2] Read safetensors header...", flush=True)
t0 = time.time()
with open(E2B_PATH, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen).decode("utf-8"))
header.pop("__metadata__", None)
print(f"    {len(header)} tensors, {time.time()-t0:.2f}s")

def raw_read(meta):
    start = 8 + hlen + meta["data_offsets"][0]
    size  = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(E2B_PATH, "rb") as f:
        f.seek(start)
        raw = f.read(size)
    return torch.frombuffer(bytearray(raw), dtype=DTYPE_MAP[meta["dtype"]]).reshape(meta["shape"])

# Find layer prefix
for prefix in ["model.language_model.layers", "language_model.layers", "model.layers"]:
    if f"{prefix}.0.mlp.gate_proj.weight" in header:
        LAYER_PREFIX = prefix
        break
# Find embed
for cand in ["model.language_model.embed_tokens.weight",
             "language_model.embed_tokens.weight",
             "model.embed_tokens.weight"]:
    if cand in header:
        EMBED_KEY = cand
        break
print(f"    layer prefix: {LAYER_PREFIX}")
print(f"    embed key: {EMBED_KEY}")


# ── 3. Load embed + unit-normalize once ───────────────────────────────
print("\n[3] Load embed_tokens, unit-normalize...", flush=True)
embed = raw_read(header[EMBED_KEY]).to(DEVICE).float()      # [vocab, hidden]
embed_norm = embed / embed.norm(dim=1, keepdim=True).clamp_min(1e-9)
vocab_size, hidden = embed.shape
print(f"    embed: {tuple(embed.shape)}, dtype={embed.dtype}")
print(f"    GPU after embed: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")


# ── 4. Collect all annotations ─────────────────────────────────────────
print("\n[4] Load annotations...", flush=True)
annotations = []   # list of dicts with: layer, neuron, source, coherence, category, gate_theme, down_theme, ...

# phase10
with open(P10_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        n = json.loads(line)
        ann = n.get("annotation", {})
        annotations.append({
            "layer":               n["layer"],
            "neuron":              n["index"],
            "source":              "phase10",
            "coherence_score":     None,
            "functional_category": ann.get("functional_category"),
            "anxiety_score":       ann.get("rlhf_anxiety_score"),
            "gate_theme":          None,
            "down_theme":          None,
            "safe_to_overwrite":   None,
            "repurpose_risk":      None,
            "notes":               ann.get("notes", "")[:200],
        })

# phase21
for fname in sorted(os.listdir(P21_CHUNKS)):
    if not fname.endswith(".json"): continue
    with open(os.path.join(P21_CHUNKS, fname), "r", encoding="utf-8") as f:
        d = json.load(f)
        for ann in d.get("annotations", []):
            try:
                L = int(ann["id"].split("L")[1].split("#")[0])
                idx = int(ann["id"].split("#")[1])
            except Exception:
                continue
            annotations.append({
                "layer":               L,
                "neuron":              idx,
                "source":              "phase21",
                "coherence_score":     ann.get("coherence_score"),
                "functional_category": None,
                "anxiety_score":       None,
                "gate_theme":          ann.get("gate_theme"),
                "down_theme":          ann.get("down_theme"),
                "safe_to_overwrite":   1 if ann.get("safe_to_overwrite") else 0,
                "repurpose_risk":      ann.get("repurpose_risk"),
                "notes":               ann.get("notes", "")[:200],
            })

# 按 layer 分组以便高效处理
layer_to_idx = defaultdict(list)
for ann in annotations:
    layer_to_idx[ann["layer"]].append(ann["neuron"])
print(f"    total annotations: {len(annotations)}")
print(f"    layers: {sorted(layer_to_idx.keys())}")


# ── 5. SQLite schema ─────────────────────────────────────────────────
print(f"\n[5] Create SQLite DB at {DB_PATH}...", flush=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
conn = sqlite3.connect(DB_PATH)
conn.executescript("""
CREATE TABLE neurons (
    layer INT, neuron INT,
    source TEXT,
    coherence_score INT,
    functional_category TEXT,
    anxiety_score INT,
    gate_theme TEXT, down_theme TEXT,
    safe_to_overwrite INT,
    repurpose_risk TEXT,
    notes TEXT,
    -- 几何指标
    focus_gate REAL,    -- top-K gate 嵌入间平均 cosine (语义内聚度)
    focus_down REAL,    -- top-K down 嵌入间平均 cosine
    align REAL,         -- cos(gate_top_K 中心向量, down_top_K 中心向量)
    gate_norm REAL,
    down_norm REAL,
    -- top tokens (用于人工检查)
    gate_top10 TEXT,    -- "tok1|tok2|...|tok10"
    down_top10 TEXT,
    PRIMARY KEY (layer, neuron)
);
CREATE INDEX idx_source ON neurons(source);
CREATE INDEX idx_coh ON neurons(coherence_score);
CREATE INDEX idx_cat ON neurons(functional_category);
""")
conn.commit()


# ── 6. 算指标 (逐层) ─────────────────────────────────────────────────
print(f"\n[6] Compute indicators per layer...", flush=True)

def compute_focus_and_align(gate_proj_row, down_proj_row):
    """对单个神经元算 focus_gate, focus_down, align
    gate_proj_row: [vocab] float, W_gate[i] @ embed.T
    down_proj_row: [vocab] float, W_down[:,i] @ embed.T
    """
    _, gate_top_idx = gate_proj_row.topk(K)
    _, down_top_idx = down_proj_row.topk(K)

    gate_emb = embed_norm[gate_top_idx]      # [K, hidden] unit norm
    down_emb = embed_norm[down_top_idx]      # [K, hidden]

    # focus = top-K embedding 之间的平均成对余弦相似度 (排除对角)
    gate_cos_mat = gate_emb @ gate_emb.T     # [K, K]
    focus_gate = ((gate_cos_mat.sum() - K) / (K * (K - 1))).item()
    down_cos_mat = down_emb @ down_emb.T
    focus_down = ((down_cos_mat.sum() - K) / (K * (K - 1))).item()

    # align = gate 中心 vs down 中心方向余弦
    center_gate = gate_emb.mean(dim=0)
    center_gate_n = center_gate / center_gate.norm().clamp_min(1e-9)
    center_down = down_emb.mean(dim=0)
    center_down_n = center_down / center_down.norm().clamp_min(1e-9)
    align = (center_gate_n * center_down_n).sum().item()

    return focus_gate, focus_down, align, gate_top_idx.tolist(), down_top_idx.tolist()


# 提前把每个 (layer, neuron) 的 annotation 索引化, 方便查
ann_by_key = {(a["layer"], a["neuron"]): a for a in annotations}

total = 0
for L in sorted(layer_to_idx.keys()):
    neuron_idxs = sorted(set(layer_to_idx[L]))   # 唯一化
    t1 = time.time()
    # 加载 W_gate, W_down (只需对应行/列)
    gate_key = f"{LAYER_PREFIX}.{L}.mlp.gate_proj.weight"
    down_key = f"{LAYER_PREFIX}.{L}.mlp.down_proj.weight"
    if gate_key not in header:
        print(f"  L{L:02d}: SKIP (key not found)", flush=True)
        continue
    W_gate = raw_read(header[gate_key]).to(DEVICE)       # [interm, hidden]
    W_down = raw_read(header[down_key]).to(DEVICE)       # [hidden, interm]

    # 选出我们要的行/列
    idx_tensor = torch.tensor(neuron_idxs, device=DEVICE)
    gate_rows = W_gate.index_select(0, idx_tensor).float()                    # [n, hidden]
    down_cols = W_down.index_select(1, idx_tensor).T.contiguous().float()     # [n, hidden]

    # 整批投影
    gate_proj = gate_rows @ embed.T   # [n, vocab]
    down_proj = down_cols @ embed.T   # [n, vocab]

    # norms
    gate_norms = gate_rows.norm(dim=1)
    down_norms = down_cols.norm(dim=1)

    # 逐神经元算指标 + 写 db
    rows_to_insert = []
    for j, idx in enumerate(neuron_idxs):
        focus_g, focus_d, align, gate_top, down_top = compute_focus_and_align(gate_proj[j], down_proj[j])
        a = ann_by_key[(L, idx)]
        gate_top10 = "|".join(tok.decode([t]) for t in gate_top[:10])
        down_top10 = "|".join(tok.decode([t]) for t in down_top[:10])
        rows_to_insert.append((
            L, idx,
            a["source"], a["coherence_score"], a["functional_category"], a["anxiety_score"],
            a["gate_theme"], a["down_theme"],
            a["safe_to_overwrite"], a["repurpose_risk"], a["notes"],
            focus_g, focus_d, align,
            gate_norms[j].item(), down_norms[j].item(),
            gate_top10, down_top10,
        ))

    conn.executemany("""
        INSERT OR IGNORE INTO neurons VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, rows_to_insert)
    conn.commit()

    total += len(rows_to_insert)
    print(f"  L{L:02d}: {len(neuron_idxs):4d} neurons   ({time.time()-t1:.1f}s)", flush=True)
    del W_gate, W_down, gate_rows, down_cols, gate_proj, down_proj
    torch.cuda.empty_cache()

print(f"\n[6] Inserted {total} rows into DB")


# ── 7. SQL 分析: 各类标注神经元的指标分布 ────────────────────────────
print(f"\n[7] SQL group analysis:")
cur = conn.cursor()

def print_query(title, sql):
    print(f"\n  {title}")
    print(f"  {'-'*88}")
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    print(f"  " + "  ".join(f"{c:>15s}" for c in cols))
    for row in cur.fetchall():
        cells = []
        for c in row:
            if c is None:
                cells.append("    --")
            elif isinstance(c, float):
                cells.append(f"{c:15.4f}")
            elif isinstance(c, int):
                cells.append(f"{c:15d}")
            else:
                s = str(c)[:15]
                cells.append(f"{s:>15s}")
        print("  " + "  ".join(cells))

# Q1: phase10 vs phase21 by coherence_score buckets
print_query("Q1: phase10 (functional) vs phase21 by coherence bucket — mean indicators",
"""
SELECT
    CASE
        WHEN source = 'phase10' THEN 'p10_functional'
        WHEN coherence_score >= 9 THEN 'p21_coh_9-10'
        WHEN coherence_score >= 7 THEN 'p21_coh_7-8'
        WHEN coherence_score >= 4 THEN 'p21_coh_4-6'
        WHEN coherence_score >= 2 THEN 'p21_coh_2-3'
        ELSE 'p21_coh_0-1'
    END AS grp,
    COUNT(*) AS n,
    ROUND(AVG(focus_gate), 4) AS focus_gate,
    ROUND(AVG(focus_down), 4) AS focus_down,
    ROUND(AVG(align), 4) AS align_mean,
    ROUND(AVG(down_norm), 3) AS down_norm
FROM neurons
GROUP BY grp
ORDER BY grp;
""")

# Q2: phase10 by functional_category
print_query("Q2: phase10 (30) by functional_category — mean indicators",
"""
SELECT
    functional_category AS cat,
    COUNT(*) AS n,
    ROUND(AVG(focus_gate), 4) AS focus_gate,
    ROUND(AVG(focus_down), 4) AS focus_down,
    ROUND(AVG(align), 4) AS align_mean,
    ROUND(AVG(down_norm), 3) AS down_norm
FROM neurons
WHERE source = 'phase10'
GROUP BY functional_category
ORDER BY n DESC;
""")

# Q3: high coherence + high align (potential knowledge candidates in phase21)
print_query("Q3: phase21 coherence>=7 AND align>=0.2 (potential knowledge in inert pool)",
"""
SELECT layer, neuron, coherence_score,
       ROUND(focus_gate, 3) AS focus_gate,
       ROUND(focus_down, 3) AS focus_down,
       ROUND(align, 3) AS align,
       SUBSTR(gate_theme, 1, 20) AS gate_theme,
       SUBSTR(down_theme, 1, 20) AS down_theme
FROM neurons
WHERE source = 'phase21' AND coherence_score >= 7 AND align >= 0.2
ORDER BY align DESC
LIMIT 15;
""")

# Q4: phase10 spotlight — 看 5 个特别神经元的指标
print_query("Q4: spotlight on 5 famous phase10 neurons",
"""
SELECT layer, neuron, functional_category,
       ROUND(focus_gate, 3) AS focus_gate,
       ROUND(focus_down, 3) AS focus_down,
       ROUND(align, 3) AS align,
       ROUND(down_norm, 3) AS down_norm,
       SUBSTR(gate_top10, 1, 40) AS gate_top,
       SUBSTR(down_top10, 1, 40) AS down_top
FROM neurons
WHERE (layer = 26 AND neuron IN (10136, 12271, 449))
   OR (layer = 27 AND neuron IN (4115))
   OR (layer = 25 AND neuron IN (10791));
""")

# Q5: phase21 coherence=0 (truly dead/noise) baseline
print_query("Q5: phase21 coherence=0 (incoherent/noise baseline) — mean indicators",
"""
SELECT 'p21_coh_0' AS grp, COUNT(*) AS n,
       ROUND(AVG(focus_gate), 4) AS focus_gate,
       ROUND(AVG(focus_down), 4) AS focus_down,
       ROUND(AVG(align), 4) AS align_mean,
       ROUND(AVG(down_norm), 3) AS down_norm
FROM neurons
WHERE source = 'phase21' AND coherence_score = 0;
""")

conn.close()
print(f"\n[Done]  DB: {DB_PATH}", flush=True)
