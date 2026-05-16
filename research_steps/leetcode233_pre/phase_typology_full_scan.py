"""
phase_typology_full_scan.py — Step A: E2B 全 35 层 33.7 万神经元分类

复用 v2 的 raw_read + 三维指标算法, 扩展到全模型.
存 sqlite: all_neurons_e2b.db, 不存 top tokens (太大), 仅存指标 + 分类.

v3 规则:
  knowledge:   focus_down >= 0.25 AND align >= 0.80
  functional:  focus_gate >= 0.20 AND focus_down  < 0.25 AND align < 0.80
  dead_noise:  focus_gate  < 0.17 AND focus_down  < 0.17
  mixed:       其他
"""
import os, sys, json, struct, time, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
import torch

E2B_PATH = r"J:\amr\models\gemma-4-E2B-it\model.safetensors"
OUT_DIR  = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b"
DB_PATH  = os.path.join(OUT_DIR, "all_neurons_e2b.db")

K          = 30
DEVICE     = "cuda:0"
CHUNK      = 256
DTYPE_MAP  = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Header ────────────────────────────────────────────────────────
print("[1] Read header...", flush=True)
t_total = time.time()
with open(E2B_PATH, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen).decode("utf-8"))
header.pop("__metadata__", None)

def raw_read(meta):
    start = 8 + hlen + meta["data_offsets"][0]
    size  = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(E2B_PATH, "rb") as f:
        f.seek(start); raw = f.read(size)
    return torch.frombuffer(bytearray(raw), dtype=DTYPE_MAP[meta["dtype"]]).reshape(meta["shape"])

# Detect prefix
for p in ["model.language_model.layers", "language_model.layers", "model.layers"]:
    if f"{p}.0.mlp.gate_proj.weight" in header:
        LAYER_PREFIX = p; break
for c in ["model.language_model.embed_tokens.weight",
          "language_model.embed_tokens.weight", "model.embed_tokens.weight"]:
    if c in header:
        EMBED_KEY = c; break
print(f"    {len(header)} tensors, prefix={LAYER_PREFIX}")


# ── 2. Embed + unit norm ─────────────────────────────────────────────
print("\n[2] Load embed_tokens (fp32, unit-normed)...", flush=True)
embed = raw_read(header[EMBED_KEY]).to(DEVICE).float()    # [vocab, hidden]
embed_norm = embed / embed.norm(dim=1, keepdim=True).clamp_min(1e-9)
embed_T = embed.T.contiguous()                            # [hidden, vocab]
vocab_size, hidden = embed.shape
print(f"    embed: {tuple(embed.shape)}, GPU={torch.cuda.memory_allocated()/1024**3:.2f} GiB")


# ── 3. Layer list ────────────────────────────────────────────────────
layer_indices = sorted({
    int(k.split(".layers.")[1].split(".")[0])
    for k in header if ".layers." in k and ".mlp.gate_proj.weight" in k
})
print(f"\n[3] {len(layer_indices)} layers: {layer_indices[:3]}...{layer_indices[-3:]}")


# ── 4. SQLite schema ─────────────────────────────────────────────────
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
conn = sqlite3.connect(DB_PATH)
conn.executescript("""
CREATE TABLE neurons (
    layer INT, neuron INT,
    focus_gate REAL, focus_down REAL, align REAL,
    gate_norm REAL, down_norm REAL,
    predicted_class TEXT,
    PRIMARY KEY (layer, neuron)
);
""")


# ── 5. v3 classifier (Python side) ───────────────────────────────────
def classify_v3(fg, fd, al):
    if fd >= 0.25 and al >= 0.80:
        return "knowledge"
    if fg >= 0.20 and fd < 0.25 and al < 0.80:
        return "functional"
    if fg < 0.17 and fd < 0.17:
        return "dead_noise"
    return "mixed"


# ── 6. 全层扫描 ──────────────────────────────────────────────────────
print(f"\n[6] Scan {len(layer_indices)} layers, K={K}, chunk={CHUNK}...", flush=True)
global_counts = {"knowledge": 0, "functional": 0, "dead_noise": 0, "mixed": 0}
total_neurons = 0

for L in layer_indices:
    t_layer = time.time()
    gate_key = f"{LAYER_PREFIX}.{L}.mlp.gate_proj.weight"
    down_key = f"{LAYER_PREFIX}.{L}.mlp.down_proj.weight"
    W_gate = raw_read(header[gate_key]).to(DEVICE)    # [interm, hidden]
    W_down = raw_read(header[down_key]).to(DEVICE)    # [hidden, interm]
    n_neurons = W_gate.shape[0]

    # Norms
    gate_norms = W_gate.float().norm(dim=1).cpu()     # [N]
    down_norms = W_down.float().norm(dim=0).cpu()     # [N]

    # 准备结果
    focus_gates = torch.zeros(n_neurons)
    focus_downs = torch.zeros(n_neurons)
    aligns      = torch.zeros(n_neurons)

    # Chunk loop
    for s in range(0, n_neurons, CHUNK):
        e = min(s + CHUNK, n_neurons)
        gate_rows = W_gate[s:e].float()                       # [chunk, hidden]
        down_cols = W_down[:, s:e].T.float().contiguous()     # [chunk, hidden]

        # vocab projection
        gp = gate_rows @ embed_T                              # [chunk, V]
        dp = down_cols @ embed_T

        _, gate_topk = gp.topk(K, dim=1)                      # [chunk, K]
        _, down_topk = dp.topk(K, dim=1)
        del gp, dp

        # Gather embeddings (unit-normed) for top-K
        # gate_emb: [chunk, K, hidden]
        gate_emb = embed_norm[gate_topk]                      # advanced indexing
        down_emb = embed_norm[down_topk]

        # focus = mean pairwise cosine (excl diag) = (sum(emb@emb.T) - K) / (K*(K-1))
        # emb @ emb.transpose(1,2): [chunk, K, K]
        gate_cos = torch.bmm(gate_emb, gate_emb.transpose(1, 2))     # [chunk, K, K]
        down_cos = torch.bmm(down_emb, down_emb.transpose(1, 2))
        focus_g = (gate_cos.sum(dim=(1,2)) - K) / (K*(K-1))
        focus_d = (down_cos.sum(dim=(1,2)) - K) / (K*(K-1))

        # align = cos(center_gate, center_down)
        cg = gate_emb.mean(dim=1)                                    # [chunk, hidden]
        cd = down_emb.mean(dim=1)
        cg_n = cg / cg.norm(dim=1, keepdim=True).clamp_min(1e-9)
        cd_n = cd / cd.norm(dim=1, keepdim=True).clamp_min(1e-9)
        al = (cg_n * cd_n).sum(dim=1)

        focus_gates[s:e] = focus_g.cpu()
        focus_downs[s:e] = focus_d.cpu()
        aligns[s:e]      = al.cpu()
        del gate_emb, down_emb, gate_cos, down_cos, cg, cd, cg_n, cd_n

    # Classify + insert
    rows = []
    layer_counts = {"knowledge": 0, "functional": 0, "dead_noise": 0, "mixed": 0}
    for i in range(n_neurons):
        fg = focus_gates[i].item()
        fd = focus_downs[i].item()
        al = aligns[i].item()
        cls = classify_v3(fg, fd, al)
        layer_counts[cls] += 1
        global_counts[cls] += 1
        rows.append((L, i, fg, fd, al,
                     gate_norms[i].item(), down_norms[i].item(), cls))
    conn.executemany("INSERT INTO neurons VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    total_neurons += n_neurons

    elapsed = time.time() - t_layer
    pct = {k: f"{100*v/n_neurons:5.1f}%" for k, v in layer_counts.items()}
    print(f"  L{L:02d}  N={n_neurons:5d}  "
          f"know={layer_counts['knowledge']:4d}({pct['knowledge']}) "
          f"func={layer_counts['functional']:4d}({pct['functional']}) "
          f"mixed={layer_counts['mixed']:4d}({pct['mixed']}) "
          f"dead={layer_counts['dead_noise']:5d}({pct['dead_noise']})  ({elapsed:.1f}s)",
          flush=True)

    del W_gate, W_down
    torch.cuda.empty_cache()

# 索引
print("\n[7] Build indexes...", flush=True)
conn.executescript("""
CREATE INDEX idx_class ON neurons(predicted_class);
CREATE INDEX idx_layer ON neurons(layer);
CREATE INDEX idx_fg ON neurons(focus_gate);
CREATE INDEX idx_fd ON neurons(focus_down);
CREATE INDEX idx_align ON neurons(align);
""")
conn.commit()


# ── 8. Summary ───────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  Total: {total_neurons:,} neurons in {time.time()-t_total:.1f}s")
print("  Global classification:")
for c, n in global_counts.items():
    print(f"    {c:12s}: {n:7,d}  ({100*n/total_neurons:5.2f}%)")

# Per-layer summary
cur = conn.cursor()
print(f"\n  Per-layer counts:")
print(f"  {'layer':6s}  {'total':6s}  {'know':5s}  {'func':5s}  {'mixed':5s}  {'dead':6s}")
cur.execute("""
    SELECT layer, COUNT(*) as total,
           SUM(CASE WHEN predicted_class='knowledge' THEN 1 ELSE 0 END) AS know,
           SUM(CASE WHEN predicted_class='functional' THEN 1 ELSE 0 END) AS func,
           SUM(CASE WHEN predicted_class='mixed' THEN 1 ELSE 0 END) AS mixed,
           SUM(CASE WHEN predicted_class='dead_noise' THEN 1 ELSE 0 END) AS dead
    FROM neurons GROUP BY layer ORDER BY layer;
""")
for L, total, know, func, mixed, dead in cur.fetchall():
    print(f"  L{L:02d}    {total:6d}  {know:5d}  {func:5d}  {mixed:5d}  {dead:6d}")

# 知识神经元集中在哪些层?
print("\n  Top 5 layers by knowledge count:")
cur.execute("""
    SELECT layer, COUNT(*) AS know_n
    FROM neurons WHERE predicted_class='knowledge'
    GROUP BY layer ORDER BY know_n DESC LIMIT 5;
""")
for L, n in cur.fetchall():
    print(f"    L{L:02d}: {n} knowledge neurons")

conn.close()
print(f"\n  DB: {DB_PATH}")
print("[Done]", flush=True)
