"""
Phase 15: 谁提的 '0' + 谁推高 '0'.

Trajectory (from phase 12):
  hidden_states[23] (= layer 22 output): no '0' in top-30
  hidden_states[24] (= layer 23 output): '0' rank 14 first appearance  ← 引入
  hidden_states[25] (= layer 24 output): '0' rank 1 p=1.0              ← 推高 lock-in

→ "提 '0' " = layer 23 MLP neurons
→ "推高 '0' = layer 24 MLP neurons

Attribution per layer L: contrib_i = act_L_i * (W_down[L][:, i] · lm_head['0'])

Output top neurons + concentration in each layer.
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
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase15_who_proposed_0.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5

# 两个关键层
INTRODUCER_LAYER = 23   # 0-indexed, its output = hidden_states[24] where '0' first appears
AMPLIFIER_LAYER = 24    # its output = hidden_states[25] where '0' rank=1

TOP_K = 20


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

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    print(f"[loaded body_ids] {len(body_ids)} tokens")

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    tok_idx_A, _, _ = find_marker_token_pos(tok, body_ids, marker_A)
    abs_pos_A = prompt_len + tok_idx_A
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)

    # Get '0' token id
    id_0 = tok.encode("0", add_special_tokens=False)[0]
    id_0_space = tok.encode(" 0", add_special_tokens=False)[0]
    print(f"[target tokens] '0' id={id_0}, ' 0' id={id_0_space}")
    v_0 = lm_head_W[id_0]
    v_0_space = lm_head_W[id_0_space]
    print(f"  ||lm_head['0']||={v_0.norm().item():.3f}  ||lm_head[' 0']||={v_0_space.norm().item():.3f}")

    # Capture intermediate at INTRODUCER + AMPLIFIER layers
    inter_per_layer = {}
    def make_hook(L):
        def fn(m, inputs):
            inter_per_layer[L] = inputs[0][0, abs_pos_A, :].to(torch.float32).cpu().clone()
        return fn
    pre_hooks = []
    for L in [INTRODUCER_LAYER, AMPLIFIER_LAYER]:
        pre_hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L)))
    hooks = install_clamp_hooks(layers, sel_clamp)

    print(f"\n[forward] K=30 clamp + capture intermediate at L{INTRODUCER_LAYER} + L{AMPLIFIER_LAYER}...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    finally:
        remove_hooks(hooks)
        for h in pre_hooks: h.remove()
    print(f"  done in {time.time()-t0:.1f}s")

    def attribute(L, target_name, target_vec):
        inter = inter_per_layer[L]
        W_down = layers[L].mlp.down_proj.weight.data.to(torch.float32).cpu()
        proj = target_vec @ W_down
        contrib = inter * proj
        abs_c = contrib.abs()
        top_idx = abs_c.topk(TOP_K).indices.tolist()
        total = abs_c.sum().item()
        top5_abs = sum(abs(contrib[i].item()) for i in top_idx[:5])
        top10_abs = sum(abs(contrib[i].item()) for i in top_idx[:10])
        return {
            "layer": L, "target": target_name,
            "total_abs": total,
            "top5_conc": top5_abs / total if total > 0 else 0,
            "top10_conc": top10_abs / total if total > 0 else 0,
            "top": [(i, contrib[i].item(), inter[i].item(), proj[i].item()) for i in top_idx],
        }

    print("\n" + "="*80)
    print(f"  谁提的 '0'? — Layer {INTRODUCER_LAYER} attribution to lm_head['0']")
    print("="*80)
    intro_0 = attribute(INTRODUCER_LAYER, "'0'", v_0)
    intro_0_space = attribute(INTRODUCER_LAYER, "' 0'", v_0_space)
    amp_0 = attribute(AMPLIFIER_LAYER, "'0'", v_0)
    amp_0_space = attribute(AMPLIFIER_LAYER, "' 0'", v_0_space)

    def print_attr(attr):
        print(f"\n--- L{attr['layer']:02d} → {attr['target']} ---")
        print(f"  total |contrib|={attr['total_abs']:.3f}  top-5 conc={attr['top5_conc']*100:.1f}%  "
              f"top-10 conc={attr['top10_conc']*100:.1f}%")
        print(f"  {'rank':>4} {'neuron':>13} {'contrib':>9} {'act':>8} {'proj':>10}")
        for r, (i, c, a, p) in enumerate(attr["top"], 1):
            sign = "+" if c > 0 else "-"
            print(f"    {r:>2}  L{attr['layer']:02d}#{i:<6d}{sign}  {c:+.4f}  {a:+.3f}  {p:+.5f}")

    print_attr(intro_0)
    print_attr(intro_0_space)
    print("\n" + "="*80)
    print(f"  谁推高 '0' 到 rank 1? — Layer {AMPLIFIER_LAYER} attribution")
    print("="*80)
    print_attr(amp_0)
    print_attr(amp_0_space)

    # Save
    out_lines = []
    out_lines.append("="*80)
    out_lines.append(f"  Phase 15: 谁提的 '0' + 谁推高 '0'")
    out_lines.append("="*80)
    out_lines.append(f"\nBug A position: abs_pos_A={abs_pos_A}")
    out_lines.append(f"Trajectory:")
    out_lines.append(f"  L22 output (hidden_states[23]): no '0' in top-30")
    out_lines.append(f"  L23 output (hidden_states[24]): '0' rank 14 first appearance  ← L23 MLP 提的")
    out_lines.append(f"  L24 output (hidden_states[25]): '0' rank 1 p=1.0              ← L24 MLP 推高的")
    out_lines.append("")

    for attr in [intro_0, intro_0_space, amp_0, amp_0_space]:
        out_lines.append(f"\n--- L{attr['layer']:02d} attribution → {attr['target']} ---")
        out_lines.append(f"  total |contrib|={attr['total_abs']:.3f}  "
                         f"top-5 conc={attr['top5_conc']*100:.1f}%  "
                         f"top-10 conc={attr['top10_conc']*100:.1f}%")
        for r, (i, c, a, p) in enumerate(attr["top"], 1):
            out_lines.append(f"  #{r:>2}  L{attr['layer']:02d}#{i:<6d}  contrib={c:+.4f}  act={a:+.3f}  proj={p:+.5f}")

    OUT_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
