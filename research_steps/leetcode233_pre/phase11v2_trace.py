"""
Phase 11 v2: regenerate phase4a-style sequence inline, then trace 35 layers.

steps:
  1. K=30 clamp + model.generate (think_mode, max_new=10000), capture gen_ids
  2. full_ids = prompt + gen_ids
  3. 增量 decode 找两个 markers exact token positions:
     - Bug A: cot 中 "Contribution is " (first occurrence)
     - Bug B: code 中 "while (" (after char 15000)
  4. Forward (K=30 clamp) with output_hidden_states=True, capture all 35 layer hidden states at target positions
  5. logit_lens (final_norm + lm_head + softmax) per layer
  6. Check if Bug A target tokens (P/$/$P/$P$) or Bug B target tokens (P/p/power) appear in top-30
  7. Attribution: for hits, find which neuron contributed most
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
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase11v2_trace.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 10000
TOP_K_LENS = 30
TOP_K_NEURONS = 15


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
    for h in hooks:
        h.remove()


def find_marker_token_pos(tok, body_ids, marker_str, search_after_char=0):
    """walk body_ids, decode incrementally; find first token whose cumulative decode contains marker_str.
    Returns body-relative index of token at marker end, or -1."""
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
    lm_head_W = lm_head.weight.to(torch.float32).cpu()
    n_layers = len(layers)
    eot = tok.convert_tokens_to_ids("<turn|>")

    # K=30 clamp
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
    print(f"[prompt_len] {prompt_len}")

    # Step 1: K=30 clamp + generate
    print(f"\n[gen] K={K} clamp, max_new={MAX_NEW}...")
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
    gen_ids = out[0][prompt_len:].cpu().tolist()
    stopped_by_eot = (gen_ids and gen_ids[-1] == eot)
    print(f"  gen_len={len(gen_ids)} stopped_by_eot={stopped_by_eot}")

    body_ids = gen_ids

    # Step 2: find markers
    print("\n[locate markers]...")
    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    tok_idx_A, char_end_A, char_start_A = find_marker_token_pos(tok, body_ids, marker_A)
    print(f"  Bug A: marker found at char [{char_start_A}, {char_end_A}), token {tok_idx_A}")

    marker_B = "while ("
    # search after a position that's safely in code section. Look for "class Solution" first
    decoded_full = tok.decode(body_ids, skip_special_tokens=False)
    cs_idx = decoded_full.find("class Solution")
    print(f"  'class Solution' at char {cs_idx}")
    tok_idx_B, char_end_B, char_start_B = find_marker_token_pos(tok, body_ids, marker_B, max(0, cs_idx))
    print(f"  Bug B: marker found at char [{char_start_B}, {char_end_B}), token {tok_idx_B}")

    if tok_idx_A < 0 or tok_idx_B < 0:
        print("ERROR: marker(s) not found")
        return

    # absolute positions in full_ids
    abs_pos_A = prompt_len + tok_idx_A
    abs_pos_B = prompt_len + tok_idx_B
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    print(f"\n[full_ids] T={full_ids.shape[0]}")
    print(f"  abs_pos_A={abs_pos_A}  token={tok.decode([full_ids[abs_pos_A].item()], skip_special_tokens=False)!r}  "
          f"next={tok.decode([full_ids[abs_pos_A+1].item()], skip_special_tokens=False)!r}")
    print(f"  abs_pos_B={abs_pos_B}  token={tok.decode([full_ids[abs_pos_B].item()], skip_special_tokens=False)!r}  "
          f"next={tok.decode([full_ids[abs_pos_B+1].item()], skip_special_tokens=False)!r}")

    # Step 3: forward with K=30 clamp, capture all hidden states + intermediate activations at target positions
    inter_at_A = [None] * n_layers
    inter_at_B = [None] * n_layers
    def make_pre_hook(L):
        def fn(m, inputs):
            x = inputs[0]
            inter_at_A[L] = x[0, abs_pos_A, :].to(torch.float32).cpu().clone()
            inter_at_B[L] = x[0, abs_pos_B, :].to(torch.float32).cpu().clone()
        return fn
    pre_hooks = []
    for L in range(n_layers):
        pre_hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_pre_hook(L)))

    hooks = install_clamp_hooks(layers, sel_clamp)
    print("\n[forward] capture hidden + intermediate...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out_fwd = model(input_ids=full_ids.unsqueeze(0), use_cache=False, output_hidden_states=True)
    finally:
        remove_hooks(hooks)
        for h in pre_hooks: h.remove()
    print(f"  done in {time.time()-t0:.1f}s")

    hidden_at_A = []  # n_layers + 1 (embed + layer outputs)
    hidden_at_B = []
    for L in range(n_layers + 1):
        h = out_fwd.hidden_states[L][0]
        hidden_at_A.append(h[abs_pos_A, :].to(torch.float32).cpu().clone())
        hidden_at_B.append(h[abs_pos_B, :].to(torch.float32).cpu().clone())
    del out_fwd
    torch.cuda.empty_cache()

    # Step 4: define target tokens
    cands_A = {
        "$":    tok.encode("$",    add_special_tokens=False)[0],
        "P":    tok.encode("P",    add_special_tokens=False)[0],
        " P":   tok.encode(" P",   add_special_tokens=False)[0],
        " $P$": tok.encode(" $P$", add_special_tokens=False)[0],
    }
    cands_B = {
        "P":      tok.encode("P",      add_special_tokens=False)[0],
        " P":     tok.encode(" P",     add_special_tokens=False)[0],
        "power":  tok.encode("power",  add_special_tokens=False)[0],
        " power": tok.encode(" power", add_special_tokens=False)[0],
    }
    print(f"\n[cands A] {cands_A}")
    print(f"[cands B] {cands_B}")

    def logit_lens(h):
        with torch.no_grad():
            h_bf16 = h.unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            h_normed = final_norm(h_bf16)
            logits = lm_head(h_normed).squeeze().float().cpu()
        probs = F.softmax(logits, dim=-1)
        return probs

    def scan(hidden_list, cands, label):
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        # also dump top-10 per layer
        print(f"  per-layer top 8 tokens:")
        for L in range(len(hidden_list)):
            probs = logit_lens(hidden_list[L])
            top = probs.topk(10)
            row = []
            for v, ti in zip(top.values.tolist(), top.indices.tolist()):
                s = tok.decode([ti], skip_special_tokens=False)
                row.append(f"{s!r}({v:.3f})")
            # mark layer if any candidate is in top-30
            cand_hits = []
            top30 = probs.topk(TOP_K_LENS)
            top30_ids = top30.indices.tolist()
            for c_name, c_id in cands.items():
                if c_id in top30_ids:
                    rank = top30_ids.index(c_id) + 1
                    cand_hits.append(f"{c_name}@r{rank}p={top30.values[rank-1].item():.4f}")
            hit_marker = "  ★ HIT: " + ", ".join(cand_hits) if cand_hits else ""
            if L % 3 == 0 or cand_hits:   # sample + hits
                print(f"  L{L:02d}: {' | '.join(row[:5])}{hit_marker}")

        # collect all hits
        results = {c: [] for c in cands}
        for L in range(len(hidden_list)):
            probs = logit_lens(hidden_list[L])
            top30 = probs.topk(TOP_K_LENS)
            top30_ids = top30.indices.tolist()
            for c_name, c_id in cands.items():
                if c_id in top30_ids:
                    rank = top30_ids.index(c_id) + 1
                    results[c_name].append((L, rank, top30.values[rank-1].item()))
        return results

    print(f"\n>>> Bug A position: hidden at last token before 'Contribution is ' next-token")
    print(f"    Native baseline emitted: {tok.decode([full_ids[abs_pos_A+1].item()], skip_special_tokens=False)!r}")
    results_A = scan(hidden_at_A, cands_A, "Bug A: targets = $/P/ P/ $P$")

    print(f"\n>>> Bug B position: hidden at last token of 'while ('")
    print(f"    Native baseline emitted: {tok.decode([full_ids[abs_pos_B+1].item()], skip_special_tokens=False)!r}")
    results_B = scan(hidden_at_B, cands_B, "Bug B: targets = P/ P/power/ power")

    # summary
    print(f"\n\n{'='*70}\n  SUMMARY: where targets appear in top-{TOP_K_LENS}\n{'='*70}")
    for label, results, cands in [("Bug A", results_A, cands_A), ("Bug B", results_B, cands_B)]:
        for c_name, hits in results.items():
            if hits:
                hit_str = "; ".join(f"L{L:02d}@r{r}(p={p:.4f})" for L, r, p in hits)
                print(f"  {label} '{c_name}' (id={cands[c_name]}): {hit_str}")
            else:
                print(f"  {label} '{c_name}' (id={cands[c_name]}): NOT in top-{TOP_K_LENS} at any layer")

    # Attribution: for any hit, find top contributing neurons in that layer
    print(f"\n\n{'='*70}\n  Attribution top {TOP_K_NEURONS} neurons per hit\n{'='*70}")
    attribution_data = []
    for label, results, cands, inters in [
        ("BugA", results_A, cands_A, inter_at_A),
        ("BugB", results_B, cands_B, inter_at_B),
    ]:
        for c_name, hits in results.items():
            if not hits: continue
            c_id = cands[c_name]
            e_cand = lm_head_W[c_id]
            for (L, rank, prob) in hits:
                if L == 0: continue  # embed, no attribution
                actual_L = L - 1  # hidden_states[L] = layer (L-1) output
                if actual_L < 0 or inters[actual_L] is None: continue
                inter = inters[actual_L]
                W_down = layers[actual_L].mlp.down_proj.weight.data.to(torch.float32).cpu()
                proj = e_cand @ W_down
                contrib = inter * proj
                top = contrib.abs().topk(TOP_K_NEURONS)
                print(f"\n[{label} '{c_name}' at hidden_states[{L}]=layer{actual_L} output, rank={rank} p={prob:.4f}]")
                for v, n in zip(top.values.tolist(), top.indices.tolist()):
                    attribution_data.append({
                        "label": label, "cand": c_name, "layer": actual_L, "neuron": n,
                        "contrib": float(contrib[n].item()),
                        "act": float(inter[n].item()),
                        "proj": float(proj[n].item()),
                    })
                    print(f"  L{actual_L:02d}#{n:<6d}  contrib={contrib[n].item():+.4f}  "
                          f"act={inter[n].item():+.3f}  proj={proj[n].item():+.5f}")

    # save
    summary = {
        "Bug_A": {c: hits for c, hits in results_A.items()},
        "Bug_B": {c: hits for c, hits in results_B.items()},
        "attribution_top": attribution_data[:200],   # cap
        "tok_idx_A": tok_idx_A, "tok_idx_B": tok_idx_B,
        "char_start_A": char_start_A, "char_end_A": char_end_A,
        "char_start_B": char_start_B, "char_end_B": char_end_B,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ Phase 11 v2: 35-layer trace ================\n\n")
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
