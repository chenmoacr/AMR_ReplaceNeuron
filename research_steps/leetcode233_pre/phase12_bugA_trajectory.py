"""
Phase 12: Bug A position 完整 layer trajectory.

目标: 在 Bug A 位点 (cot 'Contribution is ' 之后) 35 层 hidden state, 看模型
"决策" 在每层如何 evolve. 跟踪:
  - 错误意图 cluster: '0', ' 0', 'zero', 'nothing', 'none' (model 最终选这个)
  - 正确候选 cluster: '$', 'P', '$P', '$P$', ' $P$', ' P', 'power', ' power'
  - 概念抽象: 'count', 'contribution', 'value', 'number', '10', '10^'
  - 不确定 hedging: '?', '...', 'unclear', 'depends'

Output:
  - 每层 top 12 tokens + probs
  - 每个 candidate token 在每层 rank+prob trajectory
  - First appearance, peak rank, final rank summary
  - JSON saved for downstream analysis
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
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase12_bugA_trajectory.txt"
OUT_JSON = ROOT / "outputs" / "gemma_code_GB01" / "phase12_bugA_trajectory.json"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 10000
TOP_K_DISPLAY = 12
TRACK_TOP_K = 100   # 在 top-100 内跟踪 rank


CANDIDATE_GROUPS = {
    "wrong_intent (model 选了)": [
        "0", " 0", "zero", " zero", "nothing", " nothing", "none", " none",
        "null", " null", "Zero",
    ],
    "correct_intent ($P$ cluster)": [
        "$", "$P", "$P$", " $", "$P$.", "P", " P", "p", " p",
        " $P$", "$P$.",
    ],
    "correct_alternate (power/m)": [
        "power", " power", "power_", " power_", "m", " m", "10^",
        "10", " 10",
    ],
    "concept_abstract": [
        "count", " count", "Count", " contribution", "contribution",
        "Contribution", " value", "value", "number", " number",
    ],
    "hedging_uncertainty": [
        "?", "...", " unclear", " depends", "complex", " complex",
        " unknown", "TBD",
    ],
}


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
    text_model = model.model.language_model
    layers = text_model.layers
    final_norm = text_model.norm
    lm_head = model.lm_head
    n_layers = len(layers)
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    # Step 1: K=30 clamp + gen
    print(f"\n[gen] K={K} clamp + max_new={MAX_NEW}...")
    hooks = install_clamp_hooks(layers, sel_clamp)
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=[eot],
            )
    finally:
        remove_hooks(hooks)
    print(f"  gen done in {time.time()-t0:.1f}s")
    body_ids = out[0][prompt_len:].cpu().tolist()
    stopped = (body_ids and body_ids[-1] == eot)
    print(f"  gen_len={len(body_ids)} stopped={stopped}")

    # Step 2: find Bug A marker
    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    tok_idx_A, char_end_A, char_start_A = find_marker_token_pos(tok, body_ids, marker_A)
    print(f"  Bug A marker: char [{char_start_A},{char_end_A}), token {tok_idx_A}")
    abs_pos_A = prompt_len + tok_idx_A
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    print(f"  abs_pos_A={abs_pos_A}  next token = "
          f"{tok.decode([full_ids[abs_pos_A+1].item()], skip_special_tokens=False)!r}")

    # Step 3: forward + capture hidden_states at Bug A
    hooks = install_clamp_hooks(layers, sel_clamp)
    print("\n[forward] capture all-layer hidden at Bug A position...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out_fwd = model(input_ids=full_ids.unsqueeze(0), use_cache=False, output_hidden_states=True)
    finally:
        remove_hooks(hooks)
    print(f"  done in {time.time()-t0:.1f}s")

    hidden_at_A = []
    for L in range(n_layers + 1):
        h = out_fwd.hidden_states[L][0, abs_pos_A, :].to(torch.float32).cpu().clone()
        hidden_at_A.append(h)
    del out_fwd
    torch.cuda.empty_cache()
    print(f"  captured {len(hidden_at_A)} layer hidden states")

    # Step 4: build candidate token IDs
    all_cands = {}
    for group_name, tokens in CANDIDATE_GROUPS.items():
        for t in tokens:
            ids = tok.encode(t, add_special_tokens=False)
            if not ids: continue
            first_id = ids[0]
            # store with display name
            if first_id not in all_cands:
                all_cands[first_id] = {"display": t, "group": group_name}
    print(f"\n[candidates] {len(all_cands)} unique token IDs across {len(CANDIDATE_GROUPS)} groups")

    # Step 5: logit lens per layer
    def logit_lens(h):
        with torch.no_grad():
            h_bf16 = h.unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            h_normed = final_norm(h_bf16)
            logits = lm_head(h_normed).squeeze().float().cpu()
        probs = F.softmax(logits, dim=-1)
        return probs

    # Step 6: per-layer top + candidate ranks
    per_layer_top = []  # list of dict {layer, top_K: [(tok, prob)], cand_ranks: {cand_id: {rank,prob,display,group}}}
    for L in range(n_layers + 1):
        probs = logit_lens(hidden_at_A[L])
        top = probs.topk(TRACK_TOP_K)
        top_ids = top.indices.tolist()
        top_probs = top.values.tolist()
        # top-K_DISPLAY
        top_disp = []
        for v, ti in zip(top_probs[:TOP_K_DISPLAY], top_ids[:TOP_K_DISPLAY]):
            top_disp.append({"id": ti, "tok": tok.decode([ti], skip_special_tokens=False), "p": v})
        # candidate ranks in top-TRACK_TOP_K
        cand_ranks = {}
        for cand_id, info in all_cands.items():
            if cand_id in top_ids:
                rank = top_ids.index(cand_id) + 1
                p = top_probs[rank-1]
                cand_ranks[cand_id] = {
                    "rank": rank, "prob": p,
                    "display": info["display"], "group": info["group"],
                }
        per_layer_top.append({
            "layer": L,
            "top_K": top_disp,
            "cand_ranks": cand_ranks,
        })

    # Step 7: trajectory per candidate group
    trajectory = {grp: [] for grp in CANDIDATE_GROUPS}
    for L in range(n_layers + 1):
        layer_data = per_layer_top[L]
        for cand_id, cr in layer_data["cand_ranks"].items():
            grp = cr["group"]
            trajectory[grp].append({
                "layer": L, "tok": cr["display"], "id": cand_id,
                "rank": cr["rank"], "prob": cr["prob"],
            })

    # === Build human-readable report ===
    lines = []
    lines.append("=" * 80)
    lines.append("  Phase 12: Bug A position layer-by-layer trajectory")
    lines.append("=" * 80)
    lines.append(f"K=30 clamp + max_new={MAX_NEW}, hidden captured at token {abs_pos_A} (Bug A pos)")
    lines.append(f"Native next token (what model emitted): "
                 f"{tok.decode([full_ids[abs_pos_A+1].item()], skip_special_tokens=False)!r}")
    lines.append("")
    lines.append(f"Layer count: {n_layers+1} (embed + {n_layers} layer outputs)")
    lines.append("")

    # Per-group trajectory (most informative)
    lines.append("=" * 80)
    lines.append("  TRAJECTORY: Candidate appearances per group, across 35 layers")
    lines.append("=" * 80)
    for grp_name in CANDIDATE_GROUPS:
        lines.append(f"\n[{grp_name}]")
        hits = trajectory[grp_name]
        if not hits:
            lines.append(f"  -- NO candidates in this group ever appeared in top-{TRACK_TOP_K} at any layer --")
            continue
        # Group by token
        by_tok = {}
        for h in hits:
            key = (h["id"], h["tok"])
            by_tok.setdefault(key, []).append((h["layer"], h["rank"], h["prob"]))
        for (cid, ctok), occurrences in by_tok.items():
            occurrences.sort()
            first_L = occurrences[0][0]
            best = min(occurrences, key=lambda x: x[1])   # smallest rank = closest to top
            best_L, best_R, best_P = best
            final_L, final_R, final_P = occurrences[-1]
            line = f"  '{ctok}' (id={cid}): first@L{first_L:02d} rank{occurrences[0][1]} | "
            line += f"best@L{best_L:02d} rank{best_R} p={best_P:.4f} | "
            line += f"last@L{final_L:02d} rank{final_R} p={final_P:.4f} | "
            line += f"layers={len(occurrences)}/{n_layers+1}"
            lines.append(line)
            # detailed sequence (up to first 20 occurrences)
            detail = "    seq: " + " → ".join(f"L{L:02d}(r{r}p{p:.3f})" for L, r, p in occurrences[:20])
            if len(occurrences) > 20:
                detail += f" ...+{len(occurrences)-20}"
            lines.append(detail)

    # Per-layer top tokens (sample every 3 layers + special layers)
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  PER-LAYER top {TOP_K_DISPLAY} tokens (all 35 layers)")
    lines.append("=" * 80)
    for L in range(n_layers + 1):
        layer_data = per_layer_top[L]
        toks_str = []
        for entry in layer_data["top_K"]:
            t = entry["tok"]
            # quote and condense
            if len(t) > 14:
                t = t[:13] + "…"
            t = t.replace("\n", "↵").replace("\t", "→")
            toks_str.append(f"{t!r}({entry['p']:.3f})")
        lines.append(f"L{L:02d}: {' | '.join(toks_str)}")

    for ln in lines: print(ln)

    # Save
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "abs_pos_A": abs_pos_A, "tok_idx_A": tok_idx_A,
        "marker": marker_A, "candidate_groups": CANDIDATE_GROUPS,
        "per_layer": per_layer_top,
        "trajectory_by_group": trajectory,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
