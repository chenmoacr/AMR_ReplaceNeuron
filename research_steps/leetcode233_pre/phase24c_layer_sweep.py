"""
Phase 24c: 逐层注入测试 - 找到 "downstream 清洗" 失效的层.

测试在 L=10,15,20,25,30,33,34 各注入一个 strong 神经元 push 'MEOW',
看哪层能真正改 logits[trigger_pos].

直接用 Phase 20 census 找该层最冷的神经元 (跳过 Gemini 标注),
weight overwrite 后单 forward 抓 top-3 + MEOW rank 变化.
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

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5

ALPHA = 10.0
BETA = 200.0

TEST_LAYERS = [10, 15, 20, 25, 30, 33, 34]


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


def get_coldest_neuron_at_layer(scan_pt, L, clamped_set):
    per_layer = scan_pt["per_layer"][L]
    cm = per_layer["cot_max"]
    rm = per_layer["resp_max"]
    combined = cm + rm
    sorted_idx = combined.argsort()
    for ii in sorted_idx.tolist():
        if (L, ii) not in clamped_set:
            return ii, float(cm[ii].item()), float(rm[ii].item())
    return None, None, None


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

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    trigger_pos = prompt_len + len(cut_body) - 1
    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    print(f"[seq] trigger_pos={trigger_pos}")

    target_ids = tok.encode("MEOW", add_special_tokens=False)
    target_id = target_ids[0]
    print(f"[target] 'MEOW' id={target_id}")

    # load scan
    scan = torch.load(SCAN_PT, weights_only=False)
    clamped_set = set(tuple(x) for x in scan["clamped_set"])

    # ============= 第一次 forward: capture h at every L_inject candidate + baseline logits =============
    captured_h = {}  # L -> tensor
    def make_h_hook(L):
        def fn(m, inputs):
            captured_h[L] = inputs[0][0, trigger_pos, :].to(torch.float32).cpu().clone()
        return fn
    h_hooks = []
    for L in TEST_LAYERS:
        h_hooks.append(layers[L].mlp.gate_proj.register_forward_pre_hook(make_h_hook(L)))
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[baseline forward] capturing h@gate_proj.in for layers {TEST_LAYERS}...")
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model(input_ids=pre_ids_t, use_cache=False)
    finally:
        for h in h_hooks: h.remove()
        remove_hooks(clamp_hooks)
    logits_base = out.logits[0, trigger_pos, :].to(torch.float32)
    top3_base = logits_base.topk(3)
    target_logit_base = logits_base[target_id].item()
    target_rank_base = (logits_base.argsort(descending=True) == target_id).nonzero(as_tuple=True)[0][0].item() + 1
    print(f"\n[baseline]:")
    print(f"  top-3: {[(top3_base.indices[j].item(), top3_base.values[j].item()) for j in range(3)]}")
    print(f"  target logit={target_logit_base:.3f}  rank={target_rank_base}")
    del out

    # ============= 逐层测试 =============
    print(f"\n[layer sweep] α={ALPHA}, β={BETA}")
    results = {}
    for L in TEST_LAYERS:
        I, cm_v, rm_v = get_coldest_neuron_at_layer(scan, L, clamped_set)
        h_trig = captured_h[L]
        print(f"\n--- L{L} ---")
        print(f"  picked L{L}#{I}  cot_max={cm_v:.4f}  resp_max={rm_v:.4f}  ||h||={h_trig.norm().item():.3f}")

        # backup
        gate_W = layers[L].mlp.gate_proj.weight.data
        up_W = layers[L].mlp.up_proj.weight.data
        down_W = layers[L].mlp.down_proj.weight.data
        b_g = gate_W[I, :].clone()
        b_u = up_W[I, :].clone()
        b_d = down_W[:, I].clone()

        # rewrite
        h_unit = (h_trig / (h_trig.norm() + 1e-6)).to(gate_W.dtype).to(gate_W.device)
        v_target = lm_head_W[target_id].to(torch.float32).cpu()
        v_target_unit = (v_target / (v_target.norm() + 1e-6)).to(down_W.dtype).to(down_W.device)
        gate_W[I, :] = ALPHA * h_unit
        up_W[I, :] = ALPHA * h_unit
        down_W[:, I] = BETA * v_target_unit

        # forward
        captured_act_L = {}
        def cap_act(m, inputs):
            captured_act_L["v"] = inputs[0][0, trigger_pos, I].to(torch.float32).cpu().clone().item()
            return None
        act_hook = layers[L].mlp.down_proj.register_forward_pre_hook(cap_act)
        clamp_hooks = install_clamp_hooks(layers, sel_clamp)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out_t = model(input_ids=pre_ids_t, use_cache=False)
        finally:
            act_hook.remove()
            remove_hooks(clamp_hooks)
        logits_new = out_t.logits[0, trigger_pos, :].to(torch.float32)
        top3_new = logits_new.topk(3)
        target_logit_new = logits_new[target_id].item()
        target_rank_new = (logits_new.argsort(descending=True) == target_id).nonzero(as_tuple=True)[0][0].item() + 1
        act_v = captured_act_L["v"]
        print(f"  act={act_v:.0f}  target_logit {target_logit_base:.2f} → {target_logit_new:.2f} (Δ{target_logit_new-target_logit_base:+.2f})")
        print(f"  target_rank {target_rank_base} → {target_rank_new}")
        print(f"  new top-3: {[(top3_new.indices[j].item(), top3_new.values[j].item()) for j in range(3)]}")
        print(f"  top-1 token: {tok.decode([top3_new.indices[0].item()])!r}")
        results[L] = {
            "neuron_idx": I, "act": act_v,
            "target_logit_base": target_logit_base,
            "target_logit_new": target_logit_new,
            "target_rank_base": target_rank_base,
            "target_rank_new": target_rank_new,
            "top1_new": top3_new.indices[0].item(),
            "top1_logit_new": top3_new.values[0].item(),
            "is_meow_top1": top3_new.indices[0].item() == target_id,
        }

        # restore
        gate_W[I, :] = b_g
        up_W[I, :] = b_u
        down_W[:, I] = b_d

    print(f"\n\n[summary table]")
    print(f"  {'L':>3}  {'act':>9}  {'rank_base':>10}  {'rank_new':>10}  {'logit_Δ':>9}  {'meow_top1':>10}")
    for L in TEST_LAYERS:
        r = results[L]
        flag = "✓" if r["is_meow_top1"] else "✗"
        print(f"  {L:>3}  {r['act']:>9.0f}  {r['target_rank_base']:>10}  {r['target_rank_new']:>10}  "
              f"{r['target_logit_new']-r['target_logit_base']:>+9.2f}  {flag:>10}")


if __name__ == "__main__":
    main()
