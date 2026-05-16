"""
Phase 24: 微型连续记忆测试 - 让 N 个 inert 神经元"记住"一段短续写

设计:
  1. Target: phase18 KNOWLEDGE_INJECT 的前 N=10 个 token
  2. 在 prefix (prompt + cut_body) 之后强行 force-feed inject_first_10, 用 forward 抓
     L_inject 层每个位置的 hidden state (即将进入 MLP gate_proj 的 input).
  3. 取 N 个 inert 神经元 @ L_inject, 手改它们的:
       W_gate row[I] = α * h_trigger / ||h_trigger||
       W_up   row[I] = α * h_trigger / ||h_trigger||
       W_down col[I] = β * lm_head[inject_token_i] / ||lm_head[inject_token_i]||
     使得: 在 h_trigger 处 silu(gate) * (up · h) ≈ 显著, push inject_token_i 的 logit
  4. 还原 prompt + cut_body 后 model.generate(N_token), 看是否生成 inject 序列.

可调:
  L_inject = 18           (池子 68 个 safe 在此层)
  N = 10                  (短的, 先验证机制)
  α = 2.0                 (gate/up scale)
  β = 8.0                 (down push scale)
  inertia of others: K=30 baseline clamp + 改完的 10 个 (其它 190 个池子神经元不动)
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
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase24_memory_test.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
N_MEMORY = 10            # 让多少 token 被神经元记住
L_INJECT = 18            # 在哪层种 detector neurons
ALPHA = 2.0              # gate/up scale
BETA = 8.0               # down push scale

# target inject text - 选 phase18 KNOWLEDGE_INJECT 的前 10 个 token (跨越段落开头风格)
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
    lm_head_W = model.lm_head.weight   # [vocab, hidden] bf16

    # K=30 clamp
    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    # cut_body location (same as phase18)
    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    full_decoded = tok.decode(body_ids, skip_special_tokens=False)
    marker = "We look at the digit $D$ at position $i$."
    idx = full_decoded.find(marker)
    cut_char_pos = idx + len(marker)
    cut_tok_idx = find_token_position_for_char(tok, body_ids, cut_char_pos)
    cut_body = body_ids[:cut_tok_idx + 1]

    # tokenize target inject
    target_ids = tok.encode(TARGET_INJECT_TEXT, add_special_tokens=False)
    if len(target_ids) > N_MEMORY:
        target_ids = target_ids[:N_MEMORY]
    elif len(target_ids) < N_MEMORY:
        print(f"[warn] target only {len(target_ids)} tokens, padding N_MEMORY")
        # use what we have
    N = len(target_ids)
    print(f"\n[target] N={N} tokens to memorize:")
    for i, tid in enumerate(target_ids):
        print(f"  i={i}  tok={tid}  decoded={tok.decode([tid], skip_special_tokens=False)!r}")

    # Build prompt
    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    print(f"\n[seq] prompt={prompt_len}, cut_body={len(cut_body)}")
    print(f"  prefix tail (last 80 chars): {tok.decode(cut_body[-30:], skip_special_tokens=False)[-80:]!r}")

    # ============= Step 1: capture h_trigger_i at L_INJECT input =============
    # 强行 force-feed [prompt + cut_body + target_ids] 走一遍, 捕获 L_INJECT.mlp 的 gate_proj 输入
    force_seq = prompt_ids + cut_body + target_ids
    cut_end = prompt_len + len(cut_body)   # position of first inject_token (target_ids[0])
    # neuron_i 需要在 position (cut_end + i - 1) 激活 → push token at position (cut_end + i)
    trigger_positions = [cut_end + i - 1 for i in range(N)]
    # i=0: position cut_end - 1 (last cut_body token, predicts inject_0)
    print(f"\n[trigger positions] {trigger_positions}")

    captured = {}
    def cap_hook(m, inputs):
        # inputs[0]: [B, T, hidden]  这是进 gate_proj 的 input (post-norm hidden)
        x = inputs[0][0]   # [T, hidden]
        for i, p in enumerate(trigger_positions):
            captured[i] = x[p, :].to(torch.float32).cpu().clone()
        return None  # no modification

    h_force = layers[L_INJECT].mlp.gate_proj.register_forward_pre_hook(cap_hook)
    clamp_hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[force-feed] capturing h at L{L_INJECT} for {N} trigger positions...")
    force_ids_t = torch.tensor([force_seq], dtype=torch.long).to(DEVICE)
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=force_ids_t, use_cache=False)
    finally:
        h_force.remove()
        remove_hooks(clamp_hooks)
    print(f"  done in {time.time()-t0:.1f}s")
    for i in range(min(3, N)):
        h = captured[i]
        print(f"  h_trigger[{i}] norm={h.norm().item():.3f}  shape={tuple(h.shape)}")

    # ============= Step 2: pick N inert neurons at L_INJECT =============
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    pool_at_L = [p for p in pool if p["layer"] == L_INJECT]
    print(f"\n[pool] {len(pool_at_L)} inert neurons at L{L_INJECT}")
    if len(pool_at_L) < N:
        print(f"[WARN] only {len(pool_at_L)} available, need {N}")
        return
    chosen = pool_at_L[:N]
    print(f"[chose] N={N} neurons:")
    for i, c in enumerate(chosen):
        print(f"  i={i}  {c['id']}  cot={c['cot_max']:.4f}  coh={c['ann_coherence']}")

    # ============= Step 3: backup original weights, then rewrite =============
    L = L_INJECT
    gate_W = layers[L].mlp.gate_proj.weight.data   # [intermediate, hidden]
    up_W = layers[L].mlp.up_proj.weight.data
    down_W = layers[L].mlp.down_proj.weight.data   # [hidden, intermediate]
    print(f"\n[layer L{L}] gate_W shape={tuple(gate_W.shape)}, down_W shape={tuple(down_W.shape)}")

    # backup
    backup = {}
    for i, c in enumerate(chosen):
        I = c["index"]
        backup[i] = {
            "I": I,
            "gate_row": gate_W[I, :].clone(),
            "up_row": up_W[I, :].clone(),
            "down_col": down_W[:, I].clone(),
        }

    # rewrite
    print(f"\n[rewrite] α={ALPHA}, β={BETA}")
    for i in range(N):
        c = chosen[i]
        I = c["index"]
        h_trig = captured[i].to(gate_W.dtype).to(gate_W.device)
        h_unit = h_trig / (h_trig.norm() + 1e-6)
        target_id = target_ids[i]
        v_target = lm_head_W[target_id].to(down_W.dtype).to(down_W.device)
        v_target_unit = v_target / (v_target.norm() + 1e-6)
        # set
        gate_W[I, :] = (ALPHA * h_unit).to(gate_W.dtype)
        up_W[I, :] = (ALPHA * h_unit).to(up_W.dtype)
        down_W[:, I] = (BETA * v_target_unit).to(down_W.dtype)

    # ============= Step 4: re-run prefix forward, check rank of each target token =============
    # 不带 target_ids - 让模型自己续写
    pre_ids_t = torch.tensor([prompt_ids + cut_body], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(pre_ids_t)
    print(f"\n[run] prompt + cut_body = {pre_ids_t.shape[1]} tokens, generate {N + 5}...")
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
    print(f"\n[gen] {len(gen_ids)} tokens:")
    print(f"  decoded: {gen_text!r}")
    print(f"  target : {tok.decode(target_ids, skip_special_tokens=False)!r}")

    # 逐 token 对比
    print(f"\n[compare per-token]")
    n_match = 0
    out_lines = ["="*80, "  Phase 24: 微型连续记忆测试", "="*80, ""]
    out_lines.append(f"target: {TARGET_INJECT_TEXT!r}")
    out_lines.append(f"target_ids: {target_ids}")
    out_lines.append(f"target_text: {tok.decode(target_ids, skip_special_tokens=False)!r}")
    out_lines.append(f"L_INJECT={L_INJECT}, N={N}, alpha={ALPHA}, beta={BETA}")
    out_lines.append(f"chosen neurons: {[c['id'] for c in chosen]}")
    out_lines.append(f"\nh_trigger norms: {[round(captured[i].norm().item(), 2) for i in range(N)]}")
    out_lines.append(f"\ngen_ids: {gen_ids[:N+2]}")
    out_lines.append(f"gen_text: {gen_text!r}")
    out_lines.append("\n[per-token compare]")
    for i in range(N):
        tgt_tid = target_ids[i]
        if i < len(gen_ids):
            got_tid = gen_ids[i]
            match = "✓" if got_tid == tgt_tid else "✗"
            if got_tid == tgt_tid:
                n_match += 1
            print(f"  i={i:>2}  target={tgt_tid:<8} {tok.decode([tgt_tid],skip_special_tokens=False)!r:<20} "
                  f"got={got_tid:<8} {tok.decode([got_tid],skip_special_tokens=False)!r:<20} {match}")
            out_lines.append(f"  i={i:>2}  target={tok.decode([tgt_tid],skip_special_tokens=False)!r:<20} "
                             f"got={tok.decode([got_tid],skip_special_tokens=False)!r:<20} {match}")
        else:
            print(f"  i={i}  target={tok.decode([tgt_tid],skip_special_tokens=False)!r}  got=N/A (gen too short)")
            out_lines.append(f"  i={i}  target={tok.decode([tgt_tid],skip_special_tokens=False)!r}  got=N/A")
    print(f"\n[result] {n_match}/{N} tokens matched ({100*n_match/N:.1f}%)")
    out_lines.append(f"\n[result] {n_match}/{N} matched ({100*n_match/N:.1f}%)")

    # ============= Step 5: restore weights =============
    print(f"\n[restore] original weights for {N} neurons...")
    for i, b in backup.items():
        I = b["I"]
        gate_W[I, :] = b["gate_row"]
        up_W[I, :] = b["up_row"]
        down_W[:, I] = b["down_col"]

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
