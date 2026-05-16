"""
phase_neuron_typology.py — 全 E2B 神经元类型分类

输入: J:\amr\models\gemma-4-E2B-it\model.safetensors
输出: J:\amr\amr_wtf\outputs\neuron_typology_e2b\

数学模型:
  对每个 MLP 神经元 i (in layer L):
    gate_proj_to_vocab[i] = (W_gate[L][i, :]) @ embed.T          shape: [vocab]
    down_proj_to_vocab[i] = (W_down[L][:, i]) @ embed.T          shape: [vocab]
    取 top-K=30, 算 softmax 概率 → 算两个指标:
      top1_prob: top-1 token 的归一化概率 (越高越集中, 越知识性)
      entropy:   top-K 分布熵 (越高越分散, 越功能性/噪声性)
    另外算 ||W_down[:, i]|| (神经元写入残差流的能量)

分类规则 (用全模型分布的分位数自适应):
  dead:      down_norm < P25 of all down_norms
  knowledge: gate_top1 >= P75 AND down_top1 >= P75            (读和写都集中)
  functional:gate_top1 <= P25 AND down_norm >= median_norm    (读发散但有能量)
  other:     上面三类之外的

进一步用 amr_wtf phase10 已标注的高分知识性神经元(L26#10136 digit_topic_modulator
等) 做 sanity check, 确认 knowledge cluster 命中。
"""
import os, sys, time, json, struct
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn.functional as F

E2B_PATH  = r"J:\amr\models\gemma-4-E2B-it\model.safetensors"
OUT_DIR   = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b"
os.makedirs(OUT_DIR, exist_ok=True)

K          = 30
DEVICE     = "cuda:0"
CHUNK      = 256                  # 每次 256 个神经元算 vocab projection
DTYPE_MAP  = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# 已知 ground truth 神经元 (用于 sanity check)
GT_KNOWLEDGE = [
    ("L26#10136", 26, 10136, "digit_topic_modulator"),
    ("L27#4115",  27, 4115,  "overflow_anxiety"),
    ("L25#10791", 25, 10791, "distributed_digit_suppressor"),
    ("L26#449",   26, 449,   "uncertainty_injector"),
    ("L26#12271", 26, 12271, "algorithm_DP_suppressor"),
]


# ── 1. 读 header ──────────────────────────────────────────────────────
print("[1] Read safetensors header...", flush=True)
t0 = time.time()
with open(E2B_PATH, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen).decode("utf-8"))
header.pop("__metadata__", None)
print(f"    {len(header)} tensors, header size {hlen:,} bytes, {time.time()-t0:.2f}s", flush=True)

def raw_read(meta):
    start = 8 + hlen + meta["data_offsets"][0]
    size  = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(E2B_PATH, "rb") as f:
        f.seek(start)
        raw = f.read(size)
    return torch.frombuffer(bytearray(raw), dtype=DTYPE_MAP[meta["dtype"]]).reshape(meta["shape"])


# ── 2. 找 embed_tokens 路径 (E2B 用 language_model.embed_tokens) ─────
embed_key = None
for cand in ["model.language_model.embed_tokens.weight",
             "language_model.embed_tokens.weight",
             "model.embed_tokens.weight"]:
    if cand in header:
        embed_key = cand
        break
assert embed_key, "Cannot find embed_tokens in header"
print(f"\n[2] Load embed_tokens: {embed_key}")
embed = raw_read(header[embed_key]).to(DEVICE)        # [vocab, hidden]
vocab_size, hidden = embed.shape
print(f"    embed shape: {tuple(embed.shape)}, dtype: {embed.dtype}", flush=True)


# ── 3. 找 layer 数 + gate/down key 模板 ───────────────────────────────
layer_indices = set()
for k in header:
    if ".layers." in k and ".mlp.gate_proj.weight" in k:
        try:
            li = int(k.split(".layers.")[1].split(".")[0])
            layer_indices.add(li)
        except Exception:
            pass
layer_indices = sorted(layer_indices)
print(f"\n[3] Found {len(layer_indices)} layers: {layer_indices[:5]}..{layer_indices[-3:]}")

# 试探 key 前缀
sample_li = layer_indices[0]
for prefix in ["model.language_model.layers", "language_model.layers", "model.layers"]:
    if f"{prefix}.{sample_li}.mlp.gate_proj.weight" in header:
        LAYER_PREFIX = prefix
        break
print(f"    layer key prefix: {LAYER_PREFIX}")


# ── 4. 逐层处理 ──────────────────────────────────────────────────────
print(f"\n[4] Iterate {len(layer_indices)} layers, K={K} top tokens...", flush=True)
all_results = []   # per-layer arrays

# embed 转 fp32 用于精确投影 (262144*1536*4 = 1.6 GB GPU)
embed_t = embed.float().T.contiguous()       # [hidden, vocab]

for L in layer_indices:
    t_layer = time.time()
    gate_key = f"{LAYER_PREFIX}.{L}.mlp.gate_proj.weight"
    down_key = f"{LAYER_PREFIX}.{L}.mlp.down_proj.weight"
    W_gate = raw_read(header[gate_key]).to(DEVICE)    # [intermediate, hidden]
    W_down = raw_read(header[down_key]).to(DEVICE)    # [hidden, intermediate]
    n_neurons, _ = W_gate.shape
    assert W_down.shape == (hidden, n_neurons), f"down shape mismatch L{L}"

    # Norms (cheap, BF16 ok)
    gate_norm = W_gate.float().norm(dim=1).cpu()      # [N]
    down_norm = W_down.float().norm(dim=0).cpu()      # [N]

    # Top-K top1_prob + entropy
    gate_top1   = torch.zeros(n_neurons)
    down_top1   = torch.zeros(n_neurons)
    gate_ent    = torch.zeros(n_neurons)
    down_ent    = torch.zeros(n_neurons)

    for s in range(0, n_neurons, CHUNK):
        e = min(s + CHUNK, n_neurons)
        # Gate projection
        gp = (W_gate[s:e].float() @ embed_t)              # [chunk, V]
        top_v, _ = gp.topk(K, dim=1)
        probs = F.softmax(top_v, dim=1)
        gate_top1[s:e] = probs[:, 0].cpu()
        gate_ent[s:e]  = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=1).cpu()
        del gp, top_v, probs
        # Down projection
        # W_down[:, s:e] -> [hidden, chunk], 转置: [chunk, hidden]
        dp = (W_down[:, s:e].T.float() @ embed_t)         # [chunk, V]
        top_v, _ = dp.topk(K, dim=1)
        probs = F.softmax(top_v, dim=1)
        down_top1[s:e] = probs[:, 0].cpu()
        down_ent[s:e]  = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=1).cpu()
        del dp, top_v, probs

    all_results.append({
        "layer": L,
        "n_neurons": n_neurons,
        "gate_norm":   gate_norm.tolist(),
        "down_norm":   down_norm.tolist(),
        "gate_top1":   gate_top1.tolist(),
        "down_top1":   down_top1.tolist(),
        "gate_entropy": gate_ent.tolist(),
        "down_entropy": down_ent.tolist(),
    })

    elapsed = time.time() - t_layer
    print(f"  L{L:02d}  N={n_neurons:5d}  "
          f"gate_top1 mean={gate_top1.mean():.4f}  down_top1 mean={down_top1.mean():.4f}  "
          f"down_norm mean={down_norm.mean():.3f}  ({elapsed:.1f}s)", flush=True)

    del W_gate, W_down
    torch.cuda.empty_cache()

print(f"\n[4] Scan complete, total {time.time()-t0:.1f}s", flush=True)


# ── 5. 计算全模型分布 + 分类阈值 ────────────────────────────────────
print("\n[5] Computing thresholds from global distribution...", flush=True)
all_gate_top1  = torch.cat([torch.tensor(r["gate_top1"])   for r in all_results])
all_down_top1  = torch.cat([torch.tensor(r["down_top1"])   for r in all_results])
all_down_norm  = torch.cat([torch.tensor(r["down_norm"])   for r in all_results])
all_gate_norm  = torch.cat([torch.tensor(r["gate_norm"])   for r in all_results])
all_gate_ent   = torch.cat([torch.tensor(r["gate_entropy"]) for r in all_results])
all_down_ent   = torch.cat([torch.tensor(r["down_entropy"]) for r in all_results])
total = all_gate_top1.numel()
print(f"    total neurons: {total:,}")

def percentiles(t, ps):
    return torch.quantile(t, torch.tensor(ps).float()).tolist()

ps = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
print(f"    gate_top1  percentiles {ps}: {[round(v,4) for v in percentiles(all_gate_top1, ps)]}")
print(f"    down_top1  percentiles {ps}: {[round(v,4) for v in percentiles(all_down_top1, ps)]}")
print(f"    down_norm  percentiles {ps}: {[round(v,3) for v in percentiles(all_down_norm, ps)]}")
print(f"    gate_ent   percentiles {ps}: {[round(v,3) for v in percentiles(all_gate_ent, ps)]}")
print(f"    down_ent   percentiles {ps}: {[round(v,3) for v in percentiles(all_down_ent, ps)]}")

# Thresholds (基于分位数自适应)
DEAD_NORM_THRESH    = percentiles(all_down_norm, [0.25])[0]
KNOWLEDGE_TOP1_GATE = percentiles(all_gate_top1, [0.75])[0]
KNOWLEDGE_TOP1_DOWN = percentiles(all_down_top1, [0.75])[0]
FUNCTIONAL_TOP1_GATE= percentiles(all_gate_top1, [0.25])[0]
MEDIAN_DOWN_NORM    = percentiles(all_down_norm, [0.5])[0]

print(f"\n[5] Thresholds:")
print(f"    DEAD: down_norm < {DEAD_NORM_THRESH:.4f}")
print(f"    KNOWLEDGE: gate_top1 >= {KNOWLEDGE_TOP1_GATE:.4f} AND down_top1 >= {KNOWLEDGE_TOP1_DOWN:.4f}")
print(f"    FUNCTIONAL: gate_top1 <= {FUNCTIONAL_TOP1_GATE:.4f} AND down_norm >= {MEDIAN_DOWN_NORM:.4f}")
print(f"    OTHER: rest")


# ── 6. 分类 ─────────────────────────────────────────────────────────
print("\n[6] Classifying all neurons...", flush=True)
def classify_neuron(dn, gt1, dt1):
    if dn < DEAD_NORM_THRESH:
        return "dead"
    if gt1 >= KNOWLEDGE_TOP1_GATE and dt1 >= KNOWLEDGE_TOP1_DOWN:
        return "knowledge"
    if gt1 <= FUNCTIONAL_TOP1_GATE and dn >= MEDIAN_DOWN_NORM:
        return "functional"
    return "other"

counts_global = {"dead": 0, "knowledge": 0, "functional": 0, "other": 0}
counts_per_layer = {}
classifications = []           # list of (layer, neuron, class, gate_top1, down_top1, down_norm)
for r in all_results:
    L = r["layer"]
    layer_counts = {"dead": 0, "knowledge": 0, "functional": 0, "other": 0}
    for i in range(r["n_neurons"]):
        c = classify_neuron(r["down_norm"][i], r["gate_top1"][i], r["down_top1"][i])
        counts_global[c] += 1
        layer_counts[c] += 1
        classifications.append((L, i, c,
                                round(r["gate_top1"][i], 4),
                                round(r["down_top1"][i], 4),
                                round(r["down_norm"][i], 4)))
    counts_per_layer[L] = layer_counts

print("\n[6] Global counts:")
for k, v in counts_global.items():
    pct = v / total * 100
    print(f"    {k:10s}: {v:6,}  ({pct:5.2f}%)")

print(f"    TOTAL     : {total:,}")


# ── 7. Ground truth sanity check ─────────────────────────────────────
print("\n[7] Sanity check on known ground-truth knowledge neurons:")
gt_lookup = {(L, i): None for _, L, i, _ in GT_KNOWLEDGE}
for L_idx, i_idx, c, gt1, dt1, dn in classifications:
    if (L_idx, i_idx) in gt_lookup:
        gt_lookup[(L_idx, i_idx)] = (c, gt1, dt1, dn)

for name, L, i, cat in GT_KNOWLEDGE:
    res = gt_lookup.get((L, i))
    if res:
        c, gt1, dt1, dn = res
        flag = "✓" if c == "knowledge" else "✗"
        print(f"    {name:12s} ({cat:30s}) -> {c:10s} gate_top1={gt1:.3f}  down_top1={dt1:.3f}  down_norm={dn:.3f}  {flag}")
    else:
        print(f"    {name:12s} -> NOT FOUND (layer {L} index {i})")


# ── 8. 保存结果 ──────────────────────────────────────────────────────
print("\n[8] Saving outputs...", flush=True)
# 全模型分类清单 (CSV, 大约 33.7 万行)
csv_path = os.path.join(OUT_DIR, "classification.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("layer,neuron,class,gate_top1,down_top1,down_norm\n")
    for L, i, c, gt1, dt1, dn in classifications:
        f.write(f"{L},{i},{c},{gt1},{dt1},{dn}\n")
print(f"    {csv_path}")

# 汇总 JSON
summary = {
    "model": "Gemma 4 E2B-it",
    "total_neurons": total,
    "K_topk": K,
    "thresholds": {
        "DEAD_NORM_THRESH": DEAD_NORM_THRESH,
        "KNOWLEDGE_TOP1_GATE": KNOWLEDGE_TOP1_GATE,
        "KNOWLEDGE_TOP1_DOWN": KNOWLEDGE_TOP1_DOWN,
        "FUNCTIONAL_TOP1_GATE": FUNCTIONAL_TOP1_GATE,
        "MEDIAN_DOWN_NORM": MEDIAN_DOWN_NORM,
    },
    "counts_global": counts_global,
    "counts_per_layer": counts_per_layer,
    "gt_sanity_check": [
        {
            "name": name, "category": cat,
            "layer": L, "neuron": i,
            "predicted_class": gt_lookup.get((L, i), (None,None,None,None))[0]
        }
        for name, L, i, cat in GT_KNOWLEDGE
    ],
}
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"    {summary_path}")

# 全部指标 (用于后续重新分类 / 画图)
indicators_path = os.path.join(OUT_DIR, "raw_indicators.json")
with open(indicators_path, "w", encoding="utf-8") as f:
    json.dump([{"layer": r["layer"],
                "gate_top1": r["gate_top1"],
                "down_top1": r["down_top1"],
                "down_norm": r["down_norm"],
                "gate_norm": r["gate_norm"]} for r in all_results], f)
print(f"    {indicators_path}")

print("\n[Done]", flush=True)
