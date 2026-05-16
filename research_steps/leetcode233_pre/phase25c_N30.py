"""
Phase 25b: 给 W_gate 加 whitening / contrastive 提升 selectivity.

策略:
  W_gate[I_i] = α * (h_i - mean_others) / ||...||
  让每个 detector "看到" h_i 时强激活, 看到 h_j (j≠i) 时弱激活.

  α 也要相应放大 (因为差分后 norm 变小).
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
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase25c_N30.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
L_INJECT = 33
N_MEMORY = 30
ALPHA = 20.0
BETA = 200.0

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
    print(f"[target] N={N}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    cut_end = prompt_len + len(cut_body)
    trigger_positions = [cut_end + i - 1 for i in range(N)]

    # Step 1: force-feed + capture h
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
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        model(input_ids=force_ids_t, use_cache=False)
    h_hook.remove()
    remove_hooks(clamp_hooks)

    H = torch.stack([captured_h[i] for i in range(N)], dim=0)   # [N, hidden]
    H_mean = H.mean(dim=0, keepdim=True)   # [1, hidden]
    H_centered = H - H_mean                # [N, hidden]
    H_centered_unit = H_centered / (H_centered.norm(dim=1, keepdim=True) + 1e-6)

    # cosine check after whitening
    cos_centered = H_centered_unit @ H_centered_unit.T
    print(f"\n[whitened cosine matrix (差分 H-mean)]")
    for i in range(N):
        row = [f"{cos_centered[i, j].item():>5.2f}" for j in range(N)]
        print(f"  i={i}: {' '.join(row)}")
    off_diag = cos_centered[torch.eye(N).bool() == False]
    print(f"  off-diag mean = {off_diag.mean().item():.3f}  (lower = better selectivity)")

    # Step 2: pick N L33 neurons
    scan = torch.load(SCAN_PT, weights_only=False)
    clamped_set = set(tuple(x) for x in scan["clamped_set"])
    chosen_indices = get_cold_neurons_at_layer(scan, L_INJECT, N, clamped_set)

    # Step 3: backup + rewrite with WHITENED detection direction
    L = L_INJECT
    gate_W = layers[L].mlp.gate_proj.weight.data
    up_W = layers[L].mlp.up_proj.weight.data
    down_W = layers[L].mlp.down_proj.weight.data
    backup = {}
    for i, I in enumerate(chosen_indices):
        backup[I] = (gate_W[I, :].clone(), up_W[I, :].clone(), down_W[:, I].clone())

    print(f"\n[rewrite] α={ALPHA}, β={BETA}  (whitened W_gate)")
    for i, I in enumerate(chosen_indices):
        # W_gate detects (h - mean), strong only when h matches h_i
        v_detect = H_centered_unit[i].to(gate_W.dtype).to(gate_W.device)
        target_id = target_ids[i]
        v_target = lm_head_W[target_id].to(torch.float32).cpu()
        v_target_unit = (v_target / (v_target.norm() + 1e-6)).to(down_W.dtype).to(down_W.device)
        gate_W[I, :] = ALPHA * v_detect
        up_W[I, :] = ALPHA * v_detect
        down_W[:, I] = BETA * v_target_unit

    # Step 4: generate
    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(pre_ids_t)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=pre_ids_t, attention_mask=attn,
            max_new_tokens=N + 5, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=[eot],
        )
    remove_hooks(clamp_hooks)
    gen_ids = out[0][pre_ids_t.shape[1]:].cpu().tolist()
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)

    # Compare
    n_match = 0
    print(f"\n[compare]")
    print(f"  target: {tok.decode(target_ids)!r}")
    print(f"  got   : {gen_text!r}")
    out_lines = ["="*80, "  Phase 25b: whitened W_gate", "="*80,
                 f"target: {TARGET_INJECT_TEXT!r}", f"target_ids: {target_ids}",
                 f"L_INJECT={L_INJECT}, α={ALPHA}, β={BETA}",
                 f"chosen indices: {chosen_indices}",
                 f"\n[whitened cosine off-diag mean]: {off_diag.mean().item():.3f}",
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

    # Restore
    for I, (g, u, d) in backup.items():
        gate_W[I, :] = g
        up_W[I, :] = u
        down_W[:, I] = d

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
