"""
Phase 25: 在 L33 用 N 个神经元 push N 个连续 token.

复用 phase24c 的关键发现: L33 注入不被 downstream 清洗.

步骤:
  1. Target: 10 token of phase18 inject 开头
  2. force-feed [prompt + cut_body + target], 抓 L33.mlp.gate_proj.in 在每个 trigger 位置的 h
  3. 选 10 个 L33 自然冷神经元 (phase20 census)
  4. 每个 i ∈ [0..N-1]:
       W_gate[I_i] = α × h_trigger_i / ||h_trigger_i||
       W_up  [I_i] = α × h_trigger_i / ||h_trigger_i||
       W_down[:, I_i] = β × lm_head[token_i] / ||lm_head[token_i]||
  5. generate from [prompt + cut_body], 看 N 个 token 是否输出

  额外: 验证 selectivity (神经元 i 是否只在 trigger 位置 i 激活, 不串扰)
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
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase25_L33_continuous.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
L_INJECT = 33
N_MEMORY = 10
ALPHA = 10.0
BETA = 200.0

TARGET_INJECT_TEXT = "        Wait — let me think carefully about"


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


def get_cold_neurons_at_layer(scan_pt, L, n, clamped_set, exclude=set()):
    per_layer = scan_pt["per_layer"][L]
    cm = per_layer["cot_max"]
    rm = per_layer["resp_max"]
    combined = cm + rm
    sorted_idx = combined.argsort()
    out = []
    for ii in sorted_idx.tolist():
        if (L, ii) in clamped_set or ii in exclude:
            continue
        out.append((ii, float(cm[ii].item()), float(rm[ii].item())))
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

    target_ids = tok.encode(TARGET_INJECT_TEXT, add_special_tokens=False)[:N_MEMORY]
    N = len(target_ids)
    print(f"\n[target] N={N} tokens:")
    for i, tid in enumerate(target_ids):
        print(f"  i={i}  {tid:<8} {tok.decode([tid])!r}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    cut_end = prompt_len + len(cut_body)
    # neuron_i fires at position (cut_end + i - 1), pushes token at position (cut_end + i)
    trigger_positions = [cut_end + i - 1 for i in range(N)]
    print(f"\n[trigger positions] {trigger_positions}")

    # Step 1: force-feed + capture h at L33 for each trigger position
    force_seq = prompt_ids + cut_body + target_ids
    force_ids_t = torch.tensor([force_seq], dtype=torch.long).to(DEVICE)
    captured_h = {}
    def cap_h_hook(m, inputs):
        x = inputs[0][0]   # [T, hidden]
        for i, p in enumerate(trigger_positions):
            captured_h[i] = x[p, :].to(torch.float32).cpu().clone()
        return None
    h_hook = layers[L_INJECT].mlp.gate_proj.register_forward_pre_hook(cap_h_hook)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[force-feed] capturing h at L{L_INJECT} for {N} positions...")
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=force_ids_t, use_cache=False)
    finally:
        h_hook.remove()
        remove_hooks(clamp_hooks)
    for i in range(min(3, N)):
        print(f"  ||h[{i}]||={captured_h[i].norm().item():.3f}")

    # cosine matrix between h_trigger pairs (selectivity check)
    print(f"\n[h_trigger cosine matrix] (should be << 1 off-diagonal for selectivity)")
    H = torch.stack([captured_h[i] for i in range(N)], dim=0)   # [N, hidden]
    H_unit = H / (H.norm(dim=1, keepdim=True) + 1e-6)
    cos_mat = H_unit @ H_unit.T   # [N, N]
    for i in range(N):
        row = [f"{cos_mat[i, j].item():>5.2f}" for j in range(N)]
        print(f"  i={i}: {' '.join(row)}")

    # Step 2: pick N cold neurons at L33
    scan = torch.load(SCAN_PT, weights_only=False)
    clamped_set = set(tuple(x) for x in scan["clamped_set"])
    cold = get_cold_neurons_at_layer(scan, L_INJECT, N, clamped_set)
    if len(cold) < N:
        print(f"ERROR: only {len(cold)} cold available")
        return
    print(f"\n[chose] N={N} L{L_INJECT} neurons:")
    for i, (I, cm_v, rm_v) in enumerate(cold):
        print(f"  i={i}  L{L_INJECT}#{I}  cot={cm_v:.4f}  resp={rm_v:.4f}")

    # Step 3: backup + rewrite
    L = L_INJECT
    gate_W = layers[L].mlp.gate_proj.weight.data
    up_W = layers[L].mlp.up_proj.weight.data
    down_W = layers[L].mlp.down_proj.weight.data
    backup = {}
    for i, (I, _, _) in enumerate(cold):
        backup[i] = {
            "I": I,
            "gate_row": gate_W[I, :].clone(),
            "up_row": up_W[I, :].clone(),
            "down_col": down_W[:, I].clone(),
        }
    print(f"\n[rewrite] α={ALPHA}, β={BETA}")
    for i in range(N):
        I, _, _ = cold[i]
        h_trig = captured_h[i].to(torch.float32)
        h_unit = (h_trig / (h_trig.norm() + 1e-6)).to(gate_W.dtype).to(gate_W.device)
        target_id = target_ids[i]
        v_target = lm_head_W[target_id].to(torch.float32).cpu()
        v_target_unit = (v_target / (v_target.norm() + 1e-6)).to(down_W.dtype).to(down_W.device)
        gate_W[I, :] = ALPHA * h_unit
        up_W[I, :] = ALPHA * h_unit
        down_W[:, I] = BETA * v_target_unit

    # Step 4: generate from prompt + cut_body
    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(pre_ids_t)
    print(f"\n[generate] {pre_ids_t.shape[1]} tokens prefix, generate {N + 5}...")
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=pre_ids_t, attention_mask=attn,
                max_new_tokens=N + 5, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=[eot],
            )
    finally:
        remove_hooks(clamp_hooks)
    gen_ids = out[0][pre_ids_t.shape[1]:].cpu().tolist()
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)

    # Compare
    n_match = 0
    print(f"\n[compare]")
    print(f"  target: {tok.decode(target_ids)!r}")
    print(f"  got   : {gen_text!r}")
    out_lines = ["="*80, f"  Phase 25: L{L_INJECT} 连续记忆 N={N}", "="*80,
                 f"target: {TARGET_INJECT_TEXT!r}", f"target_ids: {target_ids}",
                 f"L_INJECT={L_INJECT}, α={ALPHA}, β={BETA}",
                 f"chosen: {[(i, cold[i][0]) for i in range(N)]}",
                 f"\n[h cosine] off-diag mean={cos_mat[torch.eye(N).bool() == False].mean().item():.3f}",
                 "\n[per-token]"]
    for i in range(N):
        tgt = target_ids[i]
        if i < len(gen_ids):
            got = gen_ids[i]
            match = "✓" if got == tgt else "✗"
            if got == tgt:
                n_match += 1
            line = f"  i={i:>2}  target={tok.decode([tgt])!r:<20}  got={tok.decode([got])!r:<20}  {match}"
        else:
            line = f"  i={i}  target={tok.decode([tgt])!r}  got=N/A"
        print(line)
        out_lines.append(line)
    print(f"\n[result] {n_match}/{N} matched ({100*n_match/N:.1f}%)")
    out_lines.append(f"\n[result] {n_match}/{N} matched ({100*n_match/N:.1f}%)")
    out_lines.append(f"\nfull gen: {gen_text!r}")

    # Step 5: selectivity test - capture all N neurons' activations during the gen forward
    print(f"\n[selectivity test] forwarding [prompt + cut_body + target] again, capture all N neurons' acts at all trigger positions")
    chosen_indices = [c[0] for c in cold]
    captured_acts = {}  # (i, pos_idx) -> act
    def cap_acts(m, inputs):
        x = inputs[0][0]  # [T, intermediate]
        for i, I in enumerate(chosen_indices):
            for j, p in enumerate(trigger_positions):
                captured_acts[(i, j)] = x[p, I].to(torch.float32).cpu().item()
        return None
    act_hook = layers[L_INJECT].mlp.down_proj.register_forward_pre_hook(cap_acts)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=force_ids_t, use_cache=False)
    finally:
        act_hook.remove()
        remove_hooks(clamp_hooks)
    print(f"\n  neuron act per (i=neuron, j=position) — diagonal should dominate:")
    out_lines.append(f"\n[selectivity] act matrix (rows=neurons, cols=positions):")
    header = "    " + "  ".join([f"p{j}" for j in range(N)])
    print(header)
    out_lines.append(header)
    for i in range(N):
        row = [captured_acts.get((i, j), 0) for j in range(N)]
        rmax = max(abs(x) for x in row)
        marks = ["*" if (j == i and abs(v) >= 0.5 * rmax) else " " for j, v in enumerate(row)]
        line = f"  i{i}: " + "  ".join([f"{v:>4.0f}{m}" for v, m in zip(row, marks)])
        print(line)
        out_lines.append(line)

    # Restore
    for i, b in backup.items():
        I = b["I"]
        gate_W[I, :] = b["gate_row"]
        up_W[I, :] = b["up_row"]
        down_W[:, I] = b["down_col"]
    print(f"\n[restored]")

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
