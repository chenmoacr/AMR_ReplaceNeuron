"""
Phase 11: 35 层 hidden state 追踪 — 在两错位点, 检查模型是否"原生想到"了正确 token.

设计:
  1. Forward phase4a 完整 sequence with K=30 clamp (这是错的 trajectory)
  2. 用 output_hidden_states=True 抓 layer 0..34 hidden state
  3. 在 Bug A 位点 (cot "Contribution is " 之后, model 想写 "0") 和
     Bug B 位点 (code "power_of_10 = 1; ... (P)" 之后, model 想写 "long long temp_n")
     抓每一层 hidden state
  4. 每层 hidden state → final_norm + lm_head + softmax → top-30 probs (treemap-style)
  5. 看 Bug A target tokens ('$', 'P', '$P$', ' $P$', '$P', 'P$')
     和 Bug B target tokens ('while', ' while') 是否出现在任何层 top-30
  6. 若有: 该层 + 排名 + 概率; 然后 attribution 找贡献最大 neuron

输出: per-target-token per-layer 出现情况表 + 贡献 neuron 列表
"""
import os, sys, json, re
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
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase11_native_attempts.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
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


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = phase4a_full[phase4a_full.find("\n\n")+2:]

    # Find Bug A position: "Contribution is " (first occurrence in cot, before "0")
    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    idx_A_char = phase4a_text.find(marker_A) + len(marker_A)
    # Find Bug B position: in code section, "while (" 之后 model 即将写 'n' (应是 'P' / 'power')
    marker_B = "while ("
    idx_B_char = phase4a_text.find(marker_B, 15000) + len(marker_B)   # search in code area
    print(f"[Bug A char pos] {idx_A_char}")
    print(f"  tail 60: {phase4a_text[idx_A_char-60:idx_A_char]!r}")
    print(f"  next 30: {phase4a_text[idx_A_char:idx_A_char+30]!r}")
    print(f"[Bug B char pos] {idx_B_char}")
    print(f"  tail 80: {phase4a_text[idx_B_char-80:idx_B_char]!r}")
    print(f"  next 40: {phase4a_text[idx_B_char:idx_B_char+40]!r}")

    print("\n[load] tokenizer + model...")
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
    print(f"[layers] {n_layers}")

    # K=30 clamp
    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    # Build full sequence
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(phase4a_text, add_special_tokens=False)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    prompt_len = len(prompt_ids)

    # Map char positions → token positions
    # 给定 char_idx, 找 minimum token index k 使 decode(body_ids[:k]) 长度 >= char_idx
    def char_to_token_pos(char_idx):
        lo, hi = 0, len(body_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            t = tok.decode(body_ids[:mid+1], skip_special_tokens=False)
            if len(t) >= char_idx:
                hi = mid
            else:
                lo = mid + 1
        return prompt_len + lo - 1   # last token BEFORE marker emits

    tok_pos_A = char_to_token_pos(idx_A_char)
    tok_pos_B = char_to_token_pos(idx_B_char)
    print(f"\n[token pos A] {tok_pos_A} (rel {tok_pos_A-prompt_len})  "
          f"token={tok.decode([full_ids[tok_pos_A].item()], skip_special_tokens=False)!r}")
    print(f"[token pos B] {tok_pos_B} (rel {tok_pos_B-prompt_len})  "
          f"token={tok.decode([full_ids[tok_pos_B].item()], skip_special_tokens=False)!r}")

    # Forward 1: capture all layer hidden states at tok_pos_A and tok_pos_B
    hidden_at_A = [None] * (n_layers + 1)
    hidden_at_B = [None] * (n_layers + 1)
    # Also capture down_proj input (intermediate activation 12288-d) at each layer
    # at target positions, for later neuron attribution
    inter_at_A = [None] * n_layers
    inter_at_B = [None] * n_layers

    def make_pre_hook(L):
        def fn(m, inputs):
            x = inputs[0]
            inter_at_A[L] = x[0, tok_pos_A, :].to(torch.float32).cpu().clone()
            inter_at_B[L] = x[0, tok_pos_B, :].to(torch.float32).cpu().clone()
        return fn
    pre_hooks = []
    for L in range(n_layers):
        pre_hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_pre_hook(L)))

    hooks = install_clamp_hooks(layers, sel_clamp)

    print("\n[forward 1] capture per-layer hidden + intermediate activations...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model(input_ids=full_ids.unsqueeze(0), use_cache=False,
                    output_hidden_states=True)
    remove_hooks(hooks)
    for h in pre_hooks: h.remove()

    # hidden_states: tuple of (n_layers+1), [0]=embed, [i] = layer i output, [-1]=last layer output
    for L in range(n_layers + 1):
        h = out.hidden_states[L][0]   # [T, hidden]
        hidden_at_A[L] = h[tok_pos_A, :].to(torch.float32).cpu().clone()
        hidden_at_B[L] = h[tok_pos_B, :].to(torch.float32).cpu().clone()
    del out
    torch.cuda.empty_cache()

    # Build target token IDs
    candidates_A = {
        "$":    tok.encode("$",    add_special_tokens=False),
        "P":    tok.encode("P",    add_special_tokens=False),
        " P":   tok.encode(" P",   add_special_tokens=False),
        "$P":   tok.encode("$P",   add_special_tokens=False),
        "$P$":  tok.encode("$P$",  add_special_tokens=False),
        " $P$": tok.encode(" $P$", add_special_tokens=False),
    }
    # Bug B prefix end = "while (" — model 想写 "n", correct 应是 "power_of_10" / "P" / "power"
    candidates_B = {
        "P":         tok.encode("P",         add_special_tokens=False),
        " P":        tok.encode(" P",        add_special_tokens=False),
        "p":         tok.encode("p",         add_special_tokens=False),
        " p":        tok.encode(" p",        add_special_tokens=False),
        "power":     tok.encode("power",     add_special_tokens=False),
        " power":    tok.encode(" power",    add_special_tokens=False),
        "power_of":  tok.encode("power_of",  add_special_tokens=False),
        " power_of": tok.encode(" power_of", add_special_tokens=False),
    }
    # Convert to single-token id (use first sub-token if multi-token)
    def first_id(ids_list):
        return ids_list[0] if ids_list else None
    cand_A_ids = {k: first_id(v) for k, v in candidates_A.items()}
    cand_B_ids = {k: first_id(v) for k, v in candidates_B.items()}
    print(f"\n[Bug A candidate tokens] {cand_A_ids}")
    print(f"[Bug B candidate tokens] {cand_B_ids}")

    # logit_lens for each layer (cast to bf16 to match model dtype)
    def logit_lens(h):
        with torch.no_grad():
            h_bf16 = h.unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            h_normed = final_norm(h_bf16)
            logits = lm_head(h_normed).squeeze().float().cpu()
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def scan_target_appearance(hidden_per_layer, cands, label):
        """For each layer's hidden, compute logit_lens. Check candidate tokens' rank/prob in top-K_LENS."""
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        results = {}   # cand_name → list of (layer, rank, prob)
        for c_name in cands: results[c_name] = []
        for L in range(len(hidden_per_layer)):
            h = hidden_per_layer[L]
            logits, probs = logit_lens(h)
            top = probs.topk(TOP_K_LENS)
            top_ids = top.indices.tolist()
            top_probs = top.values.tolist()
            top_ids_set = set(top_ids)
            for c_name, c_id in cands.items():
                if c_id is None: continue
                if c_id in top_ids_set:
                    rank = top_ids.index(c_id) + 1
                    p = top_probs[rank-1]
                    results[c_name].append((L, rank, p))
        # report
        for c_name, hits in results.items():
            if hits:
                hit_str = ", ".join(f"L{L:02d}@rank{r}(p={p:.4f})" for L, r, p in hits)
                print(f"  '{c_name}': APPEARED in top-{TOP_K_LENS} at {len(hits)} layers:")
                print(f"      {hit_str}")
            else:
                print(f"  '{c_name}': NOT in top-{TOP_K_LENS} at any layer")
        return results

    # Bug A: hidden at tok_pos_A across 35+1 layers
    print(f"\n>>> Bug A: hidden state at 'Contribution is ' position (tok_pos={tok_pos_A})")
    print(f"    Native baseline emitted token: {tok.decode([full_ids[tok_pos_A+1].item()], skip_special_tokens=False)!r} (next after this pos)")
    results_A = scan_target_appearance(hidden_at_A, cand_A_ids, "Bug A target tokens scan")

    # Bug B
    print(f"\n>>> Bug B: hidden state at 'power_of_10 = 1; ... (P)' position (tok_pos={tok_pos_B})")
    print(f"    Native baseline emitted token: {tok.decode([full_ids[tok_pos_B+1].item()], skip_special_tokens=False)!r} (next after this pos)")
    results_B = scan_target_appearance(hidden_at_B, cand_B_ids, "Bug B target tokens scan")

    # Top-30 probs per layer (treemap-style summary) for reference
    print(f"\n\n>>> Bug A position: top-{TOP_K_LENS} tokens per layer (sample L0,5,10,15,20,25,30,34)")
    sample_layers = [0, 5, 10, 15, 20, 25, 30, 34]
    for L in sample_layers:
        h = hidden_at_A[L]
        logits, probs = logit_lens(h)
        top = probs.topk(10)
        toks_str = []
        for v, ti in zip(top.values.tolist(), top.indices.tolist()):
            s = tok.decode([ti], skip_special_tokens=False)
            toks_str.append(f"{s!r}(p={v:.3f})")
        print(f"  L{L:02d}: {toks_str[:6]}")

    print(f"\n>>> Bug B position: top-{TOP_K_LENS} tokens per layer (sample)")
    for L in sample_layers:
        h = hidden_at_B[L]
        logits, probs = logit_lens(h)
        top = probs.topk(10)
        toks_str = []
        for v, ti in zip(top.values.tolist(), top.indices.tolist()):
            s = tok.decode([ti], skip_special_tokens=False)
            toks_str.append(f"{s!r}(p={v:.3f})")
        print(f"  L{L:02d}: {toks_str[:6]}")

    # ============================================================
    # Attribution: 对每个找到 candidate 出现的 (cand, layer L), 找贡献最大 neuron
    # contribution = act_at_pos[L, i] × (W_down[L][:, i] @ lm_head_W[cand_id])
    # ============================================================
    print(f"\n\n{'='*70}\n  Attribution: 每个出现 target 的 layer 找 top {TOP_K_NEURONS} 贡献 neuron\n{'='*70}")

    def attribute(inter_per_layer, cands, results, position_label):
        out_lines = []
        for c_name, hits in results.items():
            if not hits: continue
            c_id = cands[c_name]
            e_cand = lm_head_W[c_id]   # [hidden]
            for (L, rank, prob) in hits:
                if L == 0:
                    out_lines.append(f"\n[{position_label} | candidate '{c_name}' | layer {L} | (embed layer, skip attribution)]")
                    continue
                # 对 layer L-1 (因为 hidden_states[L] 是 layer L-1 的输出)
                # Wait: hidden_states[0]=embed, hidden_states[1]=layer0 output, ..., hidden_states[i]=layer(i-1) output
                # 所以 layer L 对应 hidden_states[L+1]. 在 results 中 L 是 hidden_states index, neuron 在 layers[L-1]
                actual_layer_idx = L - 1
                if actual_layer_idx < 0 or actual_layer_idx >= len(inter_per_layer):
                    continue
                inter = inter_per_layer[actual_layer_idx]
                if inter is None: continue
                W_down = layers[actual_layer_idx].mlp.down_proj.weight.data.to(torch.float32).cpu()
                # 贡献 per neuron: act_i × (W_down[:, i] @ e_cand)
                proj = e_cand @ W_down   # [intermediate]
                contrib = inter * proj
                top = contrib.abs().topk(TOP_K_NEURONS)
                out_lines.append(f"\n[{position_label} | '{c_name}' at hidden_states[{L}] = layer{actual_layer_idx} output, rank{rank} p={prob:.4f}]")
                out_lines.append(f"  Top {TOP_K_NEURONS} contributing neurons (in layer L{actual_layer_idx:02d}):")
                for v, n in zip(top.values.tolist(), top.indices.tolist()):
                    sign = "+" if contrib[n] > 0 else "-"
                    out_lines.append(f"    L{actual_layer_idx:02d}#{n:<6d}  contrib={contrib[n].item():+.4f}  "
                                     f"act={inter[n].item():+.3f}  proj_on_cand={proj[n].item():+.5f}")
        return out_lines

    attr_A_lines = attribute(inter_at_A, cand_A_ids, results_A, "Bug A pos")
    attr_B_lines = attribute(inter_at_B, cand_B_ids, results_B, "Bug B pos")

    for ln in attr_A_lines: print(ln)
    for ln in attr_B_lines: print(ln)

    # Save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 11: 35 层 hidden state 追踪 ================\n\n")
        f.write(f"K=30 clamp on, 在 phase4a baseline trajectory 上扫描\n\n")
        f.write(f"--- Bug A position ---\n")
        f.write(f"char pos: {idx_A_char}, token pos: {tok_pos_A}\n")
        f.write(f"prefix tail: {phase4a_text[idx_A_char-80:idx_A_char]!r}\n")
        f.write(f"native next token: {tok.decode([full_ids[tok_pos_A+1].item()], skip_special_tokens=False)!r}\n\n")
        for c_name, hits in results_A.items():
            if hits:
                f.write(f"  '{c_name}' APPEARED in top-{TOP_K_LENS} at {len(hits)} layers:\n")
                for L, r, p in hits:
                    f.write(f"    hidden_states[{L:02d}] rank={r} prob={p:.4f}\n")
            else:
                f.write(f"  '{c_name}' NOT in top-{TOP_K_LENS} at any layer\n")
        f.write(f"\n--- Bug B position ---\n")
        f.write(f"char pos: {idx_B_char}, token pos: {tok_pos_B}\n")
        f.write(f"prefix tail: {phase4a_text[idx_B_char-80:idx_B_char]!r}\n")
        f.write(f"native next token: {tok.decode([full_ids[tok_pos_B+1].item()], skip_special_tokens=False)!r}\n\n")
        for c_name, hits in results_B.items():
            if hits:
                f.write(f"  '{c_name}' APPEARED in top-{TOP_K_LENS} at {len(hits)} layers:\n")
                for L, r, p in hits:
                    f.write(f"    hidden_states[{L:02d}] rank={r} prob={p:.4f}\n")
            else:
                f.write(f"  '{c_name}' NOT in top-{TOP_K_LENS} at any layer\n")
        f.write(f"\n\n--- Attribution Bug A ---\n")
        for ln in attr_A_lines: f.write(ln + "\n")
        f.write(f"\n\n--- Attribution Bug B ---\n")
        for ln in attr_B_lines: f.write(ln + "\n")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
