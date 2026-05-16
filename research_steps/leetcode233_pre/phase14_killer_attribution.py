"""
Phase 14: Killer attribution — 找 L20-L23 推 'nothing/否定' 最大贡献的 neuron.

Phase 13 揭示 abstract intent 是 cross-lingual feature direction (' nothing' / 'ไม่' /
'NotFoundError' / etc 共享一个 abstract negation direction). 所以 attribution 用
mean(lm_head[多语言否定词]) 作为 target direction, 而非单 ' nothing'.

per-layer attribution:
  v_neg = mean of lm_head[w] for w in negation_words
  for layer L in {20, 21, 22, 23}:
    act_i = hidden mlp.down_proj input at Bug A position, neuron i (12288-d)
    proj_i = W_down[L][:, i] @ v_neg
    contrib_i = act_i * proj_i
  sort by |contrib|, get top 20
  concentration = sum(|top-5|) / sum(|all|), 看是集中还是分散
  if 集中 (>60% top-5): phase 15 Killer+Inject 可行
  if 分散 (<50% top-5): 改走末层 token-decision intervention
"""
import os, sys, json, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
GEN_IDS_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_gen_ids.pt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase14_killer.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5

ATTR_LAYERS = [20, 21, 22, 23]
TOP_K = 20

# cross-lingual negation cluster (phase 13 揭示的多语言变体)
NEGATION_WORDS = [
    " nothing", "nothing", " Nothing",
    " none", "none", " None",
    " zero", " Zero", "zero",
    "0",
    "ไม่", "ไม่ได้",   # 泰语
    " 没有", "无", "没有",   # 中文
    " kein", " keine",   # 德语
    " 없음",   # 韩语
    " null", "null", " Null",
    " empty", "empty",
    "NotFoundError",
    " nicht",   # 德语 not
    " không",   # 越南语 not
    " naught",
]


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_hooks(layers, selections):
    by_layer = {}
    for it in selections:
        by_layer.setdefault(int(it["layer"]), []).append((int(it["index"]), float(it["gain"])))
    hooks = []
    for li, items in by_layer.items():
        layer_dev = layers[li].mlp.down_proj.weight.device
        idx = torch.tensor([n for n, _ in items], dtype=torch.long, device=layer_dev)
        shift_cpu = torch.tensor([s for _, s in items], dtype=torch.float32)
        cache = {}
        def make_fn(idx_local, shift_cpu_local, cache_local):
            def pre_hook(m, inputs):
                x = inputs[0]
                key = (x.dtype, x.device)
                shift_dev = cache_local.get(key)
                if shift_dev is None:
                    shift_dev = shift_cpu_local.to(dtype=x.dtype, device=x.device)
                    cache_local[key] = shift_dev
                x = x.clone()
                x[:, :, idx_local] = x[:, :, idx_local] + shift_dev
                return (x,) + inputs[1:]
            return pre_hook
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(
            make_fn(idx, shift_cpu, cache)))
    return hooks


def remove_hooks(hooks):
    for h in hooks: h.remove()


def find_marker_token_pos(tok, body_ids, marker_str, search_after_char=0):
    decoded = ""
    for i, tid in enumerate(body_ids):
        decoded += tok.decode([tid], skip_special_tokens=False)
        idx = decoded.find(marker_str, search_after_char)
        if idx >= 0:
            marker_end = idx + len(marker_str)
            if len(decoded) >= marker_end:
                return i, marker_end, idx
    return -1, -1, -1


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])

    print("[load] tokenizer + model...")
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
    layers = model.model.language_model.layers
    lm_head = model.lm_head
    lm_head_W = lm_head.weight.to(torch.float32).cpu()
    n_layers = len(layers)

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    # load cached body_ids
    if not GEN_IDS_PATH.exists():
        print(f"ERROR: {GEN_IDS_PATH} not found. Run phase13 first to generate + cache.")
        return
    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    print(f"[loaded body_ids] {len(body_ids)} tokens")

    # build sequence
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    # find Bug A
    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    tok_idx_A, _, _ = find_marker_token_pos(tok, body_ids, marker_A)
    abs_pos_A = prompt_len + tok_idx_A
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    print(f"[Bug A] abs_pos_A={abs_pos_A}, next token (Gemma emitted)="
          f"{tok.decode([full_ids[abs_pos_A+1].item()], skip_special_tokens=False)!r}")

    # build v_neg
    print(f"\n[building v_neg from {len(NEGATION_WORDS)} negation words...]")
    neg_vecs = []
    for w in NEGATION_WORDS:
        ids = tok.encode(w, add_special_tokens=False)
        if not ids:
            print(f"  skip empty: {w!r}")
            continue
        tid = ids[0]
        v = lm_head_W[tid]
        neg_vecs.append((w, tid, v))
        print(f"  '{w}' (id={tid}) norm={v.norm().item():.3f}")
    print(f"  total {len(neg_vecs)} concept vectors")
    # average
    v_neg = torch.stack([v for _, _, v in neg_vecs]).mean(0)
    v_neg_norm = v_neg.norm().item()
    print(f"  v_neg (mean) norm = {v_neg_norm:.3f}")

    # Step: forward with K=30 clamp, capture L20-23 mlp.down_proj input (= intermediate) at Bug A position
    inter_per_layer = {}
    def make_hook(L):
        def fn(m, inputs):
            x = inputs[0]
            inter_per_layer[L] = x[0, abs_pos_A, :].to(torch.float32).cpu().clone()
        return fn
    pre_hooks = []
    for L in ATTR_LAYERS:
        pre_hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L)))
    hooks = install_clamp_hooks(layers, sel_clamp)

    print("\n[forward] K=30 clamp + capture intermediate at L20-L23...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    finally:
        remove_hooks(hooks)
        for h in pre_hooks: h.remove()
    print(f"  done in {time.time()-t0:.1f}s")

    # attribution per layer
    out_lines = []
    out_lines.append("="*80)
    out_lines.append(f"  Phase 14: Killer attribution (推 'nothing/否定' direction)")
    out_lines.append("="*80)
    out_lines.append(f"v_neg = mean of lm_head[{len(neg_vecs)} negation tokens], norm={v_neg_norm:.3f}")
    out_lines.append("")

    per_layer_top = {}
    per_layer_concentration = {}

    for L in ATTR_LAYERS:
        inter = inter_per_layer[L]   # [12288]
        W_down = layers[L].mlp.down_proj.weight.data.to(torch.float32).cpu()  # [hidden, 12288]
        proj = v_neg @ W_down   # [12288]: for each neuron i, W_down[:, i] · v_neg
        contrib = inter * proj   # [12288]
        abs_contrib = contrib.abs()
        top_idx = abs_contrib.topk(TOP_K).indices.tolist()
        top_vals = [(i, contrib[i].item(), inter[i].item(), proj[i].item()) for i in top_idx]

        total_abs = abs_contrib.sum().item()
        top5_abs = sum(abs(contrib[i].item()) for i in top_idx[:5])
        concentration5 = top5_abs / total_abs if total_abs > 0 else 0.0
        top10_abs = sum(abs(contrib[i].item()) for i in top_idx[:10])
        concentration10 = top10_abs / total_abs if total_abs > 0 else 0.0

        per_layer_top[L] = top_vals
        per_layer_concentration[L] = (concentration5, concentration10, total_abs)

        out_lines.append(f"\n[L{L:02d}]  total |contrib| sum = {total_abs:.3f}  "
                         f"top-5 concentration = {concentration5*100:.1f}%  "
                         f"top-10 = {concentration10*100:.1f}%")
        out_lines.append(f"  {'rank':>4} {'neuron':>13} {'contrib':>9} {'act':>8} {'proj_on_neg':>11}")
        for r, (n, c, a, p) in enumerate(top_vals, 1):
            sign = "★" if c > 0 else "·"
            out_lines.append(f"    {r:>2}  L{L:02d}#{n:<6d}{sign}  {c:+.4f}  {a:+.3f}  {p:+.5f}")

    # global top neurons across all layers (by |contrib|)
    out_lines.append("\n" + "="*80)
    out_lines.append("  GLOBAL TOP 15 (across L20-L23, by |contrib|)")
    out_lines.append("="*80)
    all_pairs = []
    for L in ATTR_LAYERS:
        for n, c, a, p in per_layer_top[L]:
            all_pairs.append((abs(c), L, n, c, a, p))
    all_pairs.sort(reverse=True)
    for r, (_, L, n, c, a, p) in enumerate(all_pairs[:15], 1):
        sign = "★ POS push neg" if c > 0 else "· neg push neg"
        out_lines.append(f"  #{r:>2}  L{L:02d}#{n:<6d}  contrib={c:+.4f}  act={a:+.3f}  proj={p:+.5f}  ({sign})")

    # Concentration verdict
    out_lines.append("\n" + "="*80)
    out_lines.append("  CONCENTRATION VERDICT")
    out_lines.append("="*80)
    out_lines.append(f"  Layer | top-5 conc | top-10 conc | sum|contrib| | classify")
    for L in ATTR_LAYERS:
        c5, c10, total = per_layer_concentration[L]
        if c5 > 0.6:
            classify = "CONCENTRATED (>60% top-5) — Killer 集中, Phase 15 可行"
        elif c5 > 0.4:
            classify = "MIXED (40-60% top-5) — borderline"
        else:
            classify = "DISTRIBUTED (<40% top-5) — 攻 abstract intent 难, 考虑末层 token decision 干预"
        out_lines.append(f"  L{L:02d}  | {c5*100:5.1f}%  | {c10*100:5.1f}%  | {total:7.2f}  | {classify}")

    out = "\n".join(out_lines)
    print("\n" + out)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
