"""
phase_l26_knowledge_probe.py — 抽 L26 20 个 knowledge 神经元 + Gemini 标注验证

流程:
  1. SQL 从 all_neurons_e2b.db 抽 L26 predicted_class='knowledge' 中随机 20 个
  2. raw read L26 的 W_gate + W_down, 算 top-15 push / top-15 suppress
  3. 用 tokenizer decode 成字符串
  4. 调用 Gemini 2.5 Pro (复用 amr_wtf phase21 endpoint)
  5. 解析 + 打印
"""
import os, sys, json, struct, sqlite3, time, random, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
import torch

E2B_PATH = r"J:\amr\models\gemma-4-E2B-it\model.safetensors"
TOK_PATH = r"J:\amr\models\gemma-4-E2B-it"
DB_PATH  = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b\all_neurons_e2b.db"
OUT_DIR  = r"J:\amr\amr_wtf\outputs\neuron_typology_e2b"
PROBE_JSON   = os.path.join(OUT_DIR, "l26_probe_20.json")
ANNOT_JSON   = os.path.join(OUT_DIR, "l26_probe_20_annotations.json")

LAYER       = 26
N_SAMPLE    = 20
SEED        = 42
TOP_K       = 15
DEVICE      = "cuda:0"
DTYPE_MAP   = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

ENDPOINT = "http://192.168.100.1:8317"
API_KEY  = "your-api-key-1"
MODEL    = "gemini-2.5-pro"


# ── 1. SQL 随机抽样 ──────────────────────────────────────────────────
print(f"[1] Random sampling {N_SAMPLE} L{LAYER} knowledge neurons...", flush=True)
random.seed(SEED)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(f"""
    SELECT layer, neuron, focus_gate, focus_down, align, gate_norm, down_norm
    FROM neurons
    WHERE layer = ? AND predicted_class = 'knowledge'
""", (LAYER,))
all_know = cur.fetchall()
conn.close()
print(f"    L{LAYER} total knowledge: {len(all_know)}")
selected = random.sample(all_know, N_SAMPLE)
selected.sort(key=lambda r: r[1])     # 按 neuron index 排
print(f"    sampled neuron indices: {[s[1] for s in selected]}")


# ── 2. 加载 E2B header + L26 weights + embed ─────────────────────────
print(f"\n[2] Load model weights (raw read)...", flush=True)
t0 = time.time()
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
for p in ["model.language_model.layers", "language_model.layers"]:
    if f"{p}.0.mlp.gate_proj.weight" in header:
        LAYER_PREFIX = p; break
EMBED_KEY = "model.language_model.embed_tokens.weight"

embed = raw_read(header[EMBED_KEY]).to(DEVICE).float()           # [V, H]
W_gate = raw_read(header[f"{LAYER_PREFIX}.{LAYER}.mlp.gate_proj.weight"]).to(DEVICE).float()  # [I, H]
W_down = raw_read(header[f"{LAYER_PREFIX}.{LAYER}.mlp.down_proj.weight"]).to(DEVICE).float()  # [H, I]
embed_T = embed.T.contiguous()
print(f"    loaded in {time.time()-t0:.1f}s, GPU={torch.cuda.memory_allocated()/1024**3:.2f} GiB")


# ── 3. Tokenizer ─────────────────────────────────────────────────────
print(f"\n[3] Load tokenizer...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)


# ── 4. 算每个神经元的 top-15 push/suppress ───────────────────────────
print(f"\n[4] Compute top-{TOP_K} push/suppress for each...", flush=True)
indices = [s[1] for s in selected]
idx_tensor = torch.tensor(indices, device=DEVICE)
gate_rows = W_gate.index_select(0, idx_tensor)                     # [20, H]
down_cols = W_down.index_select(1, idx_tensor).T.contiguous()      # [20, H]

gate_proj = gate_rows @ embed_T                                    # [20, V]
down_proj = down_cols @ embed_T                                    # [20, V]

probe_data = []
for i, (L, n, fg, fd, al, gn, dn) in enumerate(selected):
    gp = gate_proj[i]
    dp = down_proj[i]
    gp_push_v, gp_push_i = gp.topk(TOP_K, largest=True)
    gp_supp_v, gp_supp_i = gp.topk(TOP_K, largest=False)
    dp_push_v, dp_push_i = dp.topk(TOP_K, largest=True)
    dp_supp_v, dp_supp_i = dp.topk(TOP_K, largest=False)

    def to_list(vals, idxs):
        return [{"tok": tok.decode([t.item()]), "proj": round(v.item(), 4)}
                for v, t in zip(vals, idxs)]

    probe_data.append({
        "id": f"L{L}#{n}",
        "layer": L, "neuron": n,
        "focus_gate": round(fg, 4),
        "focus_down": round(fd, 4),
        "align":      round(al, 4),
        "gate_norm":  round(gn, 4),
        "down_norm":  round(dn, 4),
        "gate_top_push":     to_list(gp_push_v, gp_push_i),
        "gate_top_suppress": to_list(gp_supp_v, gp_supp_i),
        "down_top_push":     to_list(dp_push_v, dp_push_i),
        "down_top_suppress": to_list(dp_supp_v, dp_supp_i),
    })

with open(PROBE_JSON, "w", encoding="utf-8") as f:
    json.dump(probe_data, f, ensure_ascii=False, indent=2)
print(f"    saved probe data: {PROBE_JSON}")

# 释放
del W_gate, W_down, gate_rows, down_cols, gate_proj, down_proj
torch.cuda.empty_cache()


# ── 5. 构造 Gemini prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a mechanistic interpretability assistant analyzing KNOWLEDGE NEURONS in Gemma 4 E2B-it.

CONTEXT:
We've geometrically classified neurons in this model using three indicators:
  - focus_gate = mean pairwise cosine of top-K gate-projected token embeddings
  - focus_down = mean pairwise cosine of top-K down-projected token embeddings
  - align = cos( center(gate_top_K_emb), center(down_top_K_emb) )

We call a neuron a KNOWLEDGE NEURON if BOTH gate and down project to coherent same-topic tokens (focus_down >= 0.25 AND align >= 0.80). The hypothesis is: it READS concept X and WRITES concept X back — a storage slot. We contrast this with FUNCTIONAL NEURONS that read trigger X but write unrelated operation/symbol Y.

The 20 neurons below are randomly sampled from layer L26 (the layer with the highest knowledge count, 1872 neurons = 15.2%).

YOUR TASK:
For each neuron, examine gate top-push, gate top-suppress, down top-push, down top-suppress.
Decide:
  1. gate_theme: what concept/semantic field does gate read? Use short phrase OR 'incoherent'.
  2. down_theme: what does down push or suppress?
  3. themes_match: TRUE if gate and down are in the same semantic field (cross-language synonyms count, e.g. "history" in 5 languages = same theme).
  4. stored_concept: if it IS a knowledge neuron, what concept is stored?
  5. is_real_knowledge: TRUE if you confirm it stores a coherent concept; FALSE if it looks like functional/control/noise.
  6. category: pick one — multilingual_synonym | english_concept | code_token | punctuation_pattern | functional_control | superposition_noise | other
  7. confidence: high | medium | low

CRITICAL anti-bias guidance:
- Cross-language equivalents of the SAME concept (e.g. "history"/"历史"/"Historische"/"història") still count as same-theme.
- Anti-correlation pattern (gate-push X, down-suppress X) is FUNCTIONAL behavior, not knowledge.
- If gate is clear but down is incoherent (and vice versa), it's FUNCTIONAL or NOISE, not knowledge.

Output STRICT JSON ARRAY (N=20 objects, in input order):
[
  {
    "id": "L##XXX",
    "gate_theme": "...",
    "down_theme": "...",
    "themes_match": true/false,
    "stored_concept": "...",
    "is_real_knowledge": true/false,
    "category": "...",
    "confidence": "...",
    "notes": "<1 short sentence>"
  },
  ...
]

NO markdown fences. NO preamble. Just the JSON array."""


def build_user_prompt(neurons):
    lines = [f"Annotate the {len(neurons)} L26 knowledge candidates:\n"]
    for n in neurons:
        g_push = [x["tok"] for x in n["gate_top_push"]]
        g_supp = [x["tok"] for x in n["gate_top_suppress"]]
        d_push = [x["tok"] for x in n["down_top_push"]]
        d_supp = [x["tok"] for x in n["down_top_suppress"]]
        lines.append(f"--- {n['id']} (layer {n['layer']}, neuron {n['neuron']}) ---")
        lines.append(f"  focus_gate={n['focus_gate']}  focus_down={n['focus_down']}  align={n['align']}")
        lines.append(f"  gate_top_push: {g_push}")
        lines.append(f"  gate_top_suppress: {g_supp}")
        lines.append(f"  down_top_push: {d_push}")
        lines.append(f"  down_top_suppress: {d_supp}")
        lines.append("")
    return "\n".join(lines)


# ── 6. 调用 Gemini ───────────────────────────────────────────────────
print(f"\n[5] Calling Gemini ({MODEL})...", flush=True)
user_prompt = build_user_prompt(probe_data)
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    "temperature": 0.2,
    "response_format": {"type": "json_object"},
}
req = urllib.request.Request(
    f"{ENDPOINT}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
)
t1 = time.time()
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t1
    print(f"    response in {elapsed:.1f}s, "
          f"tokens: prompt={data['usage']['prompt_tokens']}, "
          f"completion={data['usage']['completion_tokens']}")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}")
    print(f"    Probe JSON saved at {PROBE_JSON}, can manually run.")
    sys.exit(1)

content = data["choices"][0]["message"]["content"].strip()
if content.startswith("```"):
    nl = content.find("\n")
    content = content[nl+1:]
    if content.endswith("```"):
        content = content[:-3].rstrip()

parsed = json.loads(content)
if isinstance(parsed, dict):
    for k in ("neurons", "annotations", "items", "data", "result", "results"):
        if k in parsed and isinstance(parsed[k], list):
            parsed = parsed[k]; break
annotations = parsed if isinstance(parsed, list) else []
print(f"    parsed {len(annotations)} annotations")

with open(ANNOT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL, "n_neurons": len(probe_data),
        "elapsed_sec": elapsed, "usage": data.get("usage", {}),
        "annotations": annotations,
    }, f, ensure_ascii=False, indent=2)


# ── 7. 打印结果 ──────────────────────────────────────────────────────
print("\n" + "="*100)
print(f"  L{LAYER} KNOWLEDGE PROBE — Gemini Annotation Results (n={len(annotations)})")
print("="*100)

# 拼数据 + 标注
by_id = {n["id"]: n for n in probe_data}

real_know = 0
themes_match_count = 0
cat_count = {}
for i, ann in enumerate(annotations):
    nid = ann.get("id", f"#{i}")
    probe = by_id.get(nid, {})
    print(f"\n[{i+1:2d}] {nid}")
    print(f"     focus_gate={probe.get('focus_gate'):.3f}  focus_down={probe.get('focus_down'):.3f}  align={probe.get('align'):.3f}")
    print(f"     gate_theme:    {ann.get('gate_theme', '?')}")
    print(f"     down_theme:    {ann.get('down_theme', '?')}")
    print(f"     themes_match:  {ann.get('themes_match', '?')}")
    print(f"     concept:       {ann.get('stored_concept', '?')}")
    print(f"     real_knowledge:{ann.get('is_real_knowledge', '?')}   "
          f"category={ann.get('category', '?')}   conf={ann.get('confidence', '?')}")
    if ann.get("notes"):
        print(f"     notes:         {ann['notes']}")
    if probe:
        g_push = ", ".join(repr(x["tok"]) for x in probe["gate_top_push"][:6])
        d_push = ", ".join(repr(x["tok"]) for x in probe["down_top_push"][:6])
        print(f"     gate_top: {g_push}")
        print(f"     down_top: {d_push}")
    if ann.get("is_real_knowledge"): real_know += 1
    if ann.get("themes_match"):       themes_match_count += 1
    cat = ann.get("category", "?")
    cat_count[cat] = cat_count.get(cat, 0) + 1

print("\n" + "="*100)
print(f"  Summary:")
print(f"    is_real_knowledge=true: {real_know}/{len(annotations)}  ({100*real_know/max(len(annotations),1):.1f}%)")
print(f"    themes_match=true:      {themes_match_count}/{len(annotations)}")
print(f"    category distribution:")
for c, n in sorted(cat_count.items(), key=lambda x: -x[1]):
    print(f"      {c:30s}  {n}")
print(f"  Saved: {ANNOT_JSON}")
print("[Done]", flush=True)
