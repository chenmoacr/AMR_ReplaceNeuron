"""
Phase 27: 端到端 demo — 用 30 token push 修复 leetcode 233 的核心公式,
然后让 model 自由续写 200 token, 看是否输出正确 C++.

Target (~30 tokens): 简短版"D=1/D>1/D=0 三种情况下贡献是什么"
  期待: model 接续到 5. **C++ Implementation** 后写出正确 C++
        (count += A*P + (D>1 ? P : D==1 ? B+1 : 0))

  对比 baseline (无 inject): model 出错 (D>1 case 推 0 而不是 P)

成功标准:
  1. 30 token 全 match (验证机制本身)
  2. inject 后的自由续写 C++ 中包含 D>1 → +=P 的正确分支
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
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase27_end_to_end.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
L_INJECT = 33
ALPHA = 20.0
BETA = 200.0
MAX_GEN_AFTER = 600   # inject 完后再续 600 token

# 用 phase25c 已验证 N=30 100% 的同一段, 然后看后续 600 token 自由续写
TARGET_INJECT_TEXT = "        Wait — let me think carefully about what \"prefix equal to $A$\" actually means here. The prefix being fixed to $A$ does NOT mean every"


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
    # 用 phase18/25 一致的 cut 点 (前 bug 处的 marker)
    marker = "We look at the digit $D$ at position $i$."
    idx = full_decoded.find(marker)
    cut_char_pos = idx + len(marker)
    cut_tok_idx = find_token_position_for_char(tok, body_ids, cut_char_pos)
    cut_body = body_ids[:cut_tok_idx + 1]
    print(f"[cut] marker {marker!r} at char {cut_char_pos}, token {cut_tok_idx}")
    print(f"  prefix tail 100: {tok.decode(cut_body[-30:])[-100:]!r}")

    target_ids = tok.encode(TARGET_INJECT_TEXT, add_special_tokens=False)
    N = len(target_ids)
    print(f"\n[target] N={N} tokens to inject:")
    for i, tid in enumerate(target_ids):
        print(f"  i={i:>2}  {tok.decode([tid])!r}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    cut_end = prompt_len + len(cut_body)
    trigger_positions = [cut_end + i - 1 for i in range(N)]

    # ============= Step 1: force-feed & capture h =============
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
    print(f"\n[force-feed] capturing h at L{L_INJECT}...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        model(input_ids=force_ids_t, use_cache=False)
    h_hook.remove()
    remove_hooks(clamp_hooks)

    H = torch.stack([captured_h[i] for i in range(N)], dim=0)
    H_mean = H.mean(dim=0, keepdim=True)
    H_centered = H - H_mean
    H_unit = H_centered / (H_centered.norm(dim=1, keepdim=True) + 1e-6)
    cos_mat = H_unit @ H_unit.T
    n_off = N * (N-1)
    off_diag = cos_mat[torch.eye(N).bool() == False]
    print(f"\n[selectivity] off-diag mean={off_diag.mean().item():.3f}, max={off_diag.abs().max().item():.3f}")

    # ============= Step 2: pick neurons + rewrite =============
    scan = torch.load(SCAN_PT, weights_only=False)
    clamped_set = set(tuple(x) for x in scan["clamped_set"])
    chosen_indices = get_cold_neurons_at_layer(scan, L_INJECT, N, clamped_set)
    L = L_INJECT
    gate_W = layers[L].mlp.gate_proj.weight.data
    up_W = layers[L].mlp.up_proj.weight.data
    down_W = layers[L].mlp.down_proj.weight.data
    backup = {}
    for I in chosen_indices:
        backup[I] = (gate_W[I, :].clone(), up_W[I, :].clone(), down_W[:, I].clone())
    print(f"\n[rewrite] α={ALPHA}, β={BETA}, {N} neurons at L{L_INJECT}")
    for i, I in enumerate(chosen_indices):
        v_detect = H_unit[i].to(gate_W.dtype).to(gate_W.device)
        target_id = target_ids[i]
        v_target = lm_head_W[target_id].to(torch.float32).cpu()
        v_target_unit = (v_target / (v_target.norm() + 1e-6)).to(down_W.dtype).to(down_W.device)
        gate_W[I, :] = ALPHA * v_detect
        up_W[I, :] = ALPHA * v_detect
        down_W[:, I] = BETA * v_target_unit

    # ============= Step 3: 先跑 baseline (无 inject), 看 model 自己怎么续 =============
    # 需要先 restore, 跑 baseline, 再 rewrite, 跑 injected
    print(f"\n[step 3a] running BASELINE (no inject)...")
    for I, (g, u, d) in backup.items():
        gate_W[I, :] = g; up_W[I, :] = u; down_W[:, I] = d
    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(pre_ids_t)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_base = model.generate(
            input_ids=pre_ids_t, attention_mask=attn,
            max_new_tokens=N + MAX_GEN_AFTER, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=[eot],
        )
    remove_hooks(clamp_hooks)
    base_ids = out_base[0][pre_ids_t.shape[1]:].cpu().tolist()
    while base_ids and base_ids[-1] == eot:
        base_ids = base_ids[:-1]
    base_text = tok.decode(base_ids, skip_special_tokens=False)
    print(f"  baseline gen len: {len(base_ids)}")
    print(f"  baseline head 400 chars:\n  {base_text[:400]!r}")

    # ============= Step 4: 跑 injected =============
    print(f"\n[step 3b] running INJECTED (overwrite weights)...")
    for i, I in enumerate(chosen_indices):
        v_detect = H_unit[i].to(gate_W.dtype).to(gate_W.device)
        target_id = target_ids[i]
        v_target = lm_head_W[target_id].to(torch.float32).cpu()
        v_target_unit = (v_target / (v_target.norm() + 1e-6)).to(down_W.dtype).to(down_W.device)
        gate_W[I, :] = ALPHA * v_detect
        up_W[I, :] = ALPHA * v_detect
        down_W[:, I] = BETA * v_target_unit

    # 加 deactivation hook: inject N token 后自动旁路这些神经元 (避免 loop)
    # 计数 forward 次数, 第 1 次 = prefill (全部 prefix), 后续每次 = 1 个 new token
    # 第 1+N 次后开始把 chosen_indices 的 act 强制清零
    forward_count = [0]
    chosen_set_dev = torch.tensor(chosen_indices, dtype=torch.long, device=DEVICE)
    def deactivate_hook(m, inputs):
        forward_count[0] += 1
        if forward_count[0] > 1 + N:   # prefill + N inject tokens done
            x = inputs[0]
            x = x.clone()
            x[:, :, chosen_set_dev] = 0.0   # cut these neurons' activation
            return (x,) + inputs[1:]
        return None
    deact_hook = layers[L_INJECT].mlp.down_proj.register_forward_pre_hook(deactivate_hook)

    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_inj = model.generate(
            input_ids=pre_ids_t, attention_mask=attn,
            max_new_tokens=N + MAX_GEN_AFTER, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=[eot],
        )
    deact_hook.remove()
    remove_hooks(clamp_hooks)
    inj_ids = out_inj[0][pre_ids_t.shape[1]:].cpu().tolist()
    while inj_ids and inj_ids[-1] == eot:
        inj_ids = inj_ids[:-1]
    inj_text = tok.decode(inj_ids, skip_special_tokens=False)
    print(f"  injected gen len: {len(inj_ids)}")

    # ============= Step 5: 验证 — inject 部分 N tokens 对比 =============
    n_match_inject = 0
    print(f"\n[inject N={N} match check]")
    for i in range(N):
        tgt = target_ids[i]
        if i < len(inj_ids):
            got = inj_ids[i]
            match = "✓" if got == tgt else "✗"
            if got == tgt:
                n_match_inject += 1
            print(f"  i={i:>2}  target={tok.decode([tgt])!r:<18}  got={tok.decode([got])!r:<18}  {match}")
        else:
            print(f"  i={i}  N/A")
    print(f"\n[inject result] {n_match_inject}/{N} = {100*n_match_inject/N:.0f}%")

    # ============= Step 6: 续写部分 — 找 C++ Solution code =============
    import re
    print(f"\n[continuation part — what model wrote AFTER inject]")
    after_inject_text = tok.decode(inj_ids[N:], skip_special_tokens=False) if len(inj_ids) > N else ""
    print(f"  length: {len(after_inject_text)} chars")
    print(f"  head 500: {after_inject_text[:500]!r}")

    # 找 C++ class Solution
    cpp_inj = re.search(r'class Solution.*?\};', inj_text, re.DOTALL)
    cpp_base = re.search(r'class Solution.*?\};', base_text, re.DOTALL)

    print(f"\n[C++ extract — INJECTED]:")
    if cpp_inj:
        print(cpp_inj.group(0)[:1500])
    else:
        print("  (no complete C++ found)")
        print(f"  tail 600: {inj_text[-600:]!r}")

    print(f"\n[C++ extract — BASELINE for comparison]:")
    if cpp_base:
        print(cpp_base.group(0)[:1500])
    else:
        print("  (no complete C++ found)")
        print(f"  tail 600: {base_text[-600:]!r}")

    # 信号检测 (复用 phase18 风格)
    def check_signals(text, name):
        sigs = {
            "D > 1 → += P (correct)": bool(re.search(r"\bif\s*\(?\s*[Dd]\s*>\s*1\s*\)?[^}]{0,80}(\+=|=)\s*P\b", text)),
            "D == 1 (D=1 branch)":    bool(re.search(r"\bif\s*\(?\s*[Dd]\s*={1,2}\s*1\s*\)?", text)),
            "D == 0 branch (or skip)": bool(re.search(r"\bif\s*\(?\s*[Dd]\s*={1,2}\s*0\s*\)?", text)),
            "+= B + 1 (D=1 case)":    bool(re.search(r"\+=\s*B\s*\+\s*1|\+=\s*\(?\s*B\s*\+\s*1", text)),
            "P *= 10":                bool(re.search(r"\bP\s*\*=\s*10", text)),
            "count += A * P":         bool(re.search(r"count\s*\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*P", text)),
            "class Solution found":   bool(re.search(r"class Solution", text)),
            "n /= 10 (Bug B):":       bool(re.search(r"\bn\s*/=\s*10", text)),
        }
        print(f"\n  signals — {name}:")
        for k, v in sigs.items():
            print(f"    {k}: {v}")
        return sigs

    sigs_base = check_signals(base_text, "BASELINE")
    sigs_inj = check_signals(inj_text, "INJECTED")

    # ============= Step 7: 总结 =============
    out_lines = ["="*80, "  Phase 27: 端到端 demo — 30 token inject → 自由续写 C++", "="*80,
                 f"\ncut at marker {marker!r}",
                 f"target text: {TARGET_INJECT_TEXT!r}",
                 f"N={N}, L={L_INJECT}, α={ALPHA}, β={BETA}, max_gen_after={MAX_GEN_AFTER}",
                 f"selectivity cos off-diag mean={off_diag.mean().item():.3f}",
                 f"\n--- inject match: {n_match_inject}/{N} ({100*n_match_inject/N:.0f}%) ---"]
    out_lines.append(f"\n=== BASELINE (no inject) gen ===\n{base_text}")
    out_lines.append(f"\n=== INJECTED gen ===\n{inj_text}")
    out_lines.append(f"\n=== signals comparison ===")
    out_lines.append(f"  {'signal':<35}  {'baseline':>10}  {'injected':>10}")
    for k in sigs_base:
        out_lines.append(f"  {k:<35}  {str(sigs_base[k]):>10}  {str(sigs_inj[k]):>10}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")

    # restore
    for I, (g, u, d) in backup.items():
        gate_W[I, :] = g; up_W[I, :] = u; down_W[:, I] = d
    print(f"\n[restored][save] {OUT_TXT}")


if __name__ == "__main__":
    main()
