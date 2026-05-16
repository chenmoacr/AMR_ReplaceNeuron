"""
Phase 24b: 单 token push debug - 先排查 weight rewrite 是否生效.

最小测试: 在 cut_body 后, 让 1 个神经元 push token 'MEOW' 出来.
- 抓 h at cut_end - 1, L_INJECT input
- 重写一个 inert 神经元的 W_gate/W_up/W_down with EXTREME 参数 (α=10, β=200)
- forward 一次, 看 logits[cut_end-1] 的 top-3 是不是 MEOW
- 同时验证: 该神经元的 act at position cut_end-1 是不是 大
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
POOL_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase22_inert_pool_200.json"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
L_INJECT = 18

# extreme params to flush out any sign issues
ALPHA = 10.0
BETA = 200.0


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

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    cut_end = prompt_len + len(cut_body)
    trigger_pos = cut_end - 1   # last cut_body position
    print(f"[seq] prompt={prompt_len}  cut_body={len(cut_body)}  trigger_pos={trigger_pos}")

    # Target: MEOW (multi-token but use first)
    target_ids = tok.encode("MEOW", add_special_tokens=False)
    target_id = target_ids[0]
    print(f"[target] 'MEOW' first token id={target_id}  decoded={tok.decode([target_id])!r}")

    # Pick 1 inert neuron at L18
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    pool_at_L = [p for p in pool if p["layer"] == L_INJECT]
    chosen = pool_at_L[0]
    I = chosen["index"]
    print(f"[chosen neuron] L{L_INJECT}#{I}  cot_max={chosen['cot_max']:.4f}  coh={chosen['ann_coherence']}")

    # ============= Step 1: capture h_trigger, baseline logits =============
    captured_h = {}
    captured_act_orig = {}
    def cap_h_hook(m, inputs):
        captured_h["h"] = inputs[0][0, trigger_pos, :].to(torch.float32).cpu().clone()
        return None
    def cap_act_orig_pre_down(m, inputs):
        captured_act_orig["act"] = inputs[0][0, trigger_pos, I].to(torch.float32).cpu().clone().item()
        return None

    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    h_hook = layers[L_INJECT].mlp.gate_proj.register_forward_pre_hook(cap_h_hook)
    act_hook = layers[L_INJECT].mlp.down_proj.register_forward_pre_hook(cap_act_orig_pre_down)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[baseline forward] capturing h@L{L_INJECT}.in and act of L{L_INJECT}#{I}...")
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model(input_ids=pre_ids_t, use_cache=False)
    finally:
        h_hook.remove()
        act_hook.remove()
        remove_hooks(clamp_hooks)

    h_trigger = captured_h["h"]
    act_orig = captured_act_orig["act"]
    print(f"  ||h_trigger||={h_trigger.norm().item():.3f}")
    print(f"  original act of neuron at trigger_pos: {act_orig:.6f}")

    # baseline top-3 prediction at trigger_pos
    logits_base = out.logits[0, trigger_pos, :].to(torch.float32)
    top3_base = logits_base.topk(3)
    print(f"  baseline top-3 logits at trigger_pos:")
    for j in range(3):
        tid = top3_base.indices[j].item()
        print(f"    {j+1}. {tid:<8} {tok.decode([tid])!r:<20}  logit={top3_base.values[j].item():.3f}")
    target_logit_base = logits_base[target_id].item()
    target_rank_base = (logits_base.argsort(descending=True) == target_id).nonzero(as_tuple=True)[0][0].item() + 1
    print(f"  target 'MEOW' baseline: logit={target_logit_base:.3f}, rank={target_rank_base}")
    del out

    # ============= Step 2: backup + rewrite =============
    L = L_INJECT
    gate_W = layers[L].mlp.gate_proj.weight.data
    up_W = layers[L].mlp.up_proj.weight.data
    down_W = layers[L].mlp.down_proj.weight.data

    backup_gate = gate_W[I, :].clone()
    backup_up = up_W[I, :].clone()
    backup_down = down_W[:, I].clone()

    h_unit = h_trigger / (h_trigger.norm() + 1e-6)
    v_target = lm_head_W[target_id].to(torch.float32).cpu()
    v_target_unit = v_target / (v_target.norm() + 1e-6)
    print(f"\n[rewrite] α={ALPHA}, β={BETA}")
    print(f"  ||lm_head[MEOW]||={v_target.norm().item():.3f}")

    new_gate = (ALPHA * h_unit).to(gate_W.dtype).to(gate_W.device)
    new_up = (ALPHA * h_unit).to(up_W.dtype).to(up_W.device)
    new_down = (BETA * v_target_unit).to(down_W.dtype).to(down_W.device)

    gate_W[I, :] = new_gate
    up_W[I, :] = new_up
    down_W[:, I] = new_down

    # verify the write actually happened
    print(f"  verify gate[{I}, 0:3] = {gate_W[I, 0:3].cpu().tolist()}")
    print(f"  expected: {new_gate[0:3].cpu().tolist()}")

    # ============= Step 3: re-forward and check act + logits =============
    captured_act_new = {}
    def cap_act_new(m, inputs):
        captured_act_new["act"] = inputs[0][0, trigger_pos, I].to(torch.float32).cpu().clone().item()
        return None
    act_hook = layers[L_INJECT].mlp.down_proj.register_forward_pre_hook(cap_act_new)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[rewrite forward] checking new act + top-3...")
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out2 = model(input_ids=pre_ids_t, use_cache=False)
    finally:
        act_hook.remove()
        remove_hooks(clamp_hooks)

    act_new = captured_act_new["act"]
    logits_new = out2.logits[0, trigger_pos, :].to(torch.float32)
    print(f"\n  new act of neuron at trigger_pos: {act_new:.4f}  (orig was {act_orig:.6f})")
    top3_new = logits_new.topk(3)
    print(f"  new top-3 logits at trigger_pos:")
    for j in range(3):
        tid = top3_new.indices[j].item()
        print(f"    {j+1}. {tid:<8} {tok.decode([tid])!r:<20}  logit={top3_new.values[j].item():.3f}")
    target_logit_new = logits_new[target_id].item()
    target_rank_new = (logits_new.argsort(descending=True) == target_id).nonzero(as_tuple=True)[0][0].item() + 1
    print(f"  target 'MEOW' new: logit={target_logit_new:.3f}  (was {target_logit_base:.3f}, Δ={target_logit_new-target_logit_base:+.3f}), rank={target_rank_new}  (was {target_rank_base})")

    # ============= Step 4: restore =============
    gate_W[I, :] = backup_gate
    up_W[I, :] = backup_up
    down_W[:, I] = backup_down
    print(f"\n[restored]")


if __name__ == "__main__":
    main()
