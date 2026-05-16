"""
Phase 26: sparse anchor — 只在 phase19 "rank>5" 的 hot spot 位置 inject 神经元
其余位置让 model 自由续写 (利用 phase19 71% autoregressive lock-in 发现).

设计:
  1. Target = phase 25d 那段 79 token 文本
  2. force-feed 一遍, 同时抓 (a) h_trigger at L33 每位置, (b) target_rank at 每位置
  3. 把 target_rank > 5 的位置标记为 "hot spot" — 只在这些位置插神经元
  4. rank ≤ 5 的位置 → 不插, 让 model 自由 emit
  5. generate, 对比

  假说: 因 anchor 数大幅减少 (~17 from 79), trigger pair 数大幅减少, 串扰也少,
  即使有 cos=0.89 临界对, 也能稳定 push.
"""
import os, sys, json, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
GEN_IDS_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_gen_ids.pt"
SCAN_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase20_dead_scan.pt"
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase26_sparse_anchor.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
L_INJECT = 33
ALPHA = 20.0
BETA = 200.0

# rank > THRESHOLD 才插神经元
HOT_RANK_THRESHOLD = 5

# 79 token of phase18 inject
TARGET_INJECT_TEXT = """\
        Wait — let me think carefully about what "prefix equal to $A$" actually means here. The prefix being fixed to $A$ does NOT mean every digit of $X$ is fixed. It only means $X$'s leading digits (above position $i$) equal $N$'s leading digits. The digit at position $i$ is what we are counting — we want cases"""


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_hooks(layers, selections):
    from collections import defaultdict
    by_layer = defaultdict(list)
    for it in selections:
        by_layer[int(it["layer"])].append((int(it["index"]), float(it["gain"])))
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


def find_token_position_for_char(tok, body_ids, char_target):
    decoded = ""
    for i, tid in enumerate(body_ids):
        decoded += tok.decode([tid], skip_special_tokens=False)
        if len(decoded) >= char_target:
            return i
    return -1


def get_cold_neurons_at_layer(scan_pt, L, n, clamped_set):
    per_layer = scan_pt["per_layer"][L]
    cm = per_layer["cot_max"]
    rm = per_layer["resp_max"]
    combined = cm + rm
    sorted_idx = combined.argsort()
    out = []
    for ii in sorted_idx.tolist():
        if (L, ii) in clamped_set:
            continue
        out.append(ii)
        if len(out) >= n:
            break
    return out


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
    eot = tok.convert_tokens_to_ids("<turn|>")
    lm_head_W = model.lm_head.weight

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    full_decoded = tok.decode(body_ids, skip_special_tokens=False)
    marker = "We look at the digit $D$ at position $i$."
    idx = full_decoded.find(marker)
    cut_char_pos = idx + len(marker)
    cut_tok_idx = find_token_position_for_char(tok, body_ids, cut_char_pos)
    cut_body = body_ids[:cut_tok_idx + 1]

    target_ids = tok.encode(TARGET_INJECT_TEXT, add_special_tokens=False)
    N = len(target_ids)
    print(f"[target] N={N} total tokens")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    cut_end = prompt_len + len(cut_body)
    trigger_positions = [cut_end + i - 1 for i in range(N)]

    # ============= Step 1: force-feed, 同时抓 h + target_rank 每位置 =============
    force_seq = prompt_ids + cut_body + target_ids
    force_ids_t = torch.tensor([force_seq], dtype=torch.long).to(DEVICE)
    captured_h = {}
    def cap_h_hook(m, inputs):
        x = inputs[0][0]
        for i, p in enumerate(trigger_positions):
            captured_h[i] = x[p, :].to(torch.float32).cpu().clone()
        return None
    h_hook = layers[L_INJECT].mlp.gate_proj.register_forward_pre_hook(cap_h_hook)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[force-feed] capturing h@L{L_INJECT} + computing target_rank per position...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model(input_ids=force_ids_t, use_cache=False)
    h_hook.remove()
    remove_hooks(clamp_hooks)
    logits = out.logits[0]  # [T, V]
    target_ranks = []
    for i, p in enumerate(trigger_positions):
        target_id = target_ids[i]
        l_row = logits[p].to(torch.float32)
        rank = (l_row.argsort(descending=True) == target_id).nonzero(as_tuple=True)[0][0].item() + 1
        target_ranks.append(rank)
    del out, logits

    # ============= Step 2: identify hot spots =============
    hot_idx = [i for i, r in enumerate(target_ranks) if r > HOT_RANK_THRESHOLD]
    free_idx = [i for i, r in enumerate(target_ranks) if r <= HOT_RANK_THRESHOLD]
    print(f"\n[hot spots] N={N}, threshold rank>{HOT_RANK_THRESHOLD}")
    print(f"  hot count : {len(hot_idx)}")
    print(f"  free count: {len(free_idx)}")
    print(f"\n[per-position ranks]:")
    for i in range(N):
        flag = "★" if i in hot_idx else " "
        tok_str = tok.decode([target_ids[i]])
        print(f"  i={i:>2}{flag}  rank={target_ranks[i]:>6}  target={tok_str!r}")

    # ============= Step 3: pick len(hot_idx) cold neurons at L33 =============
    scan = torch.load(SCAN_PT, weights_only=False)
    clamped_set = set(tuple(x) for x in scan["clamped_set"])
    chosen_indices = get_cold_neurons_at_layer(scan, L_INJECT, len(hot_idx), clamped_set)

    # ============= Step 4: whitened W_gate over hot_idx only =============
    H_hot = torch.stack([captured_h[i] for i in hot_idx], dim=0)
    H_mean = H_hot.mean(dim=0, keepdim=True)
    H_centered = H_hot - H_mean
    H_centered_unit = H_centered / (H_centered.norm(dim=1, keepdim=True) + 1e-6)

    # cosine matrix of hot pairs only
    cos_mat = H_centered_unit @ H_centered_unit.T
    n_hot = len(hot_idx)
    off_diag = cos_mat[torch.eye(n_hot).bool() == False]
    print(f"\n[hot cosine matrix] off-diag mean = {off_diag.mean().item():.3f}, max = {off_diag.abs().max().item():.3f}")

    # ============= Step 5: backup + rewrite =============
    L = L_INJECT
    gate_W = layers[L].mlp.gate_proj.weight.data
    up_W = layers[L].mlp.up_proj.weight.data
    down_W = layers[L].mlp.down_proj.weight.data
    backup = {}
    for h_i, I in zip(hot_idx, chosen_indices):
        backup[I] = (gate_W[I, :].clone(), up_W[I, :].clone(), down_W[:, I].clone())

    print(f"\n[rewrite] α={ALPHA}, β={BETA}, anchors only at {len(hot_idx)} hot spots")
    for k, (h_i, I) in enumerate(zip(hot_idx, chosen_indices)):
        v_detect = H_centered_unit[k].to(gate_W.dtype).to(gate_W.device)
        target_id = target_ids[h_i]
        v_target = lm_head_W[target_id].to(torch.float32).cpu()
        v_target_unit = (v_target / (v_target.norm() + 1e-6)).to(down_W.dtype).to(down_W.device)
        gate_W[I, :] = ALPHA * v_detect
        up_W[I, :] = ALPHA * v_detect
        down_W[:, I] = BETA * v_target_unit

    # ============= Step 6: generate =============
    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(pre_ids_t)
    print(f"\n[generate] N+5={N+5} tokens...")
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        gen_out = model.generate(
            input_ids=pre_ids_t, attention_mask=attn,
            max_new_tokens=N + 5, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=[eot],
        )
    remove_hooks(clamp_hooks)
    gen_ids = gen_out[0][pre_ids_t.shape[1]:].cpu().tolist()
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)

    # Compare
    n_match = 0
    n_match_hot = 0
    n_match_free = 0
    out_lines = ["="*80, "  Phase 26: sparse anchor only at phase19 hot spots", "="*80,
                 f"target N={N}, anchors={len(hot_idx)} (rank>{HOT_RANK_THRESHOLD}), free={len(free_idx)}",
                 f"target text: {TARGET_INJECT_TEXT[:200]!r}...",
                 f"\n[anchor cos] off-diag mean = {off_diag.mean().item():.3f}, max = {off_diag.abs().max().item():.3f}",
                 "\n[per-token]"]
    print(f"\n[compare]")
    for i in range(N):
        tgt = target_ids[i]
        is_hot = i in hot_idx
        flag = "★" if is_hot else " "
        if i < len(gen_ids):
            got = gen_ids[i]
            match = "✓" if got == tgt else "✗"
            if got == tgt:
                n_match += 1
                if is_hot: n_match_hot += 1
                else: n_match_free += 1
            line = f"  i={i:>2}{flag}  rank={target_ranks[i]:>6}  target={tok.decode([tgt])!r:<20}  got={tok.decode([got])!r:<20}  {match}"
        else:
            line = f"  i={i:>2}{flag}  rank={target_ranks[i]:>6}  target={tok.decode([tgt])!r}  got=N/A"
        print(line)
        out_lines.append(line)
    print(f"\n[result]")
    print(f"  total match: {n_match}/{N} ({100*n_match/N:.1f}%)")
    print(f"  hot match  : {n_match_hot}/{len(hot_idx)}  ({100*n_match_hot/max(len(hot_idx),1):.1f}%)")
    print(f"  free match : {n_match_free}/{len(free_idx)} ({100*n_match_free/max(len(free_idx),1):.1f}%)")
    out_lines.append(f"\ntotal match: {n_match}/{N}")
    out_lines.append(f"hot match  : {n_match_hot}/{len(hot_idx)}")
    out_lines.append(f"free match : {n_match_free}/{len(free_idx)}")
    out_lines.append(f"\nfull gen: {gen_text!r}")

    # Restore
    for I, (g, u, d) in backup.items():
        gate_W[I, :] = g
        up_W[I, :] = u
        down_W[:, I] = d

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
