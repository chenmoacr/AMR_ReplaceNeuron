"""
Phase 17: self-doubt 拐点 inject — 顺着 Gemma 的 "Why is it off by" 推它 self-correct.

设计:
  1. 在 phase4a body_ids 中定位 "Why is it off by" 位置 (拐点 3, char 7455)
  2. 截 body_ids 到 trigger_pos + k (k=5 让 Gemma 进入 search mode)
  3. v_redirect = normalize(
       +1.0 * lm_head[" D"]
       +1.0 * lm_head[" greater"]
       +1.0 * lm_head[" missed"]
       +0.5 * lm_head[" P"]
       -0.5 * lm_head[" zero"]
     )
  4. forward with K=30 clamp + inject @ L_inject, last prefix pos, mag ∈ [3, 5, 10]
  5. generate from prefix (with cached KV from prefill), 2000 tokens max
  6. compare continuation vs baseline (phase4a body_ids[trigger_pos+k:])

predict: 若 inject 引导 self-correct, 续 cot 会重新讨论 D > 1 case + 改公式
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
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase17_self_doubt.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
L_INJECT = 15
K_OFFSET = 5    # how many tokens after trigger to inject at
MAX_GEN = 2000

TRIGGER = "Why is it off by"

REDIRECT_CONCEPTS = [
    (" D", 1.0),
    (" greater", 1.0),
    (" missed", 1.0),
    (" P", 0.5),
    (" zero", -0.5),
]

MAGS = [0.0, 3.0, 5.0, 10.0]   # 0 = no inject (baseline forward from cut prefix)


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


def find_token_position_for_char(tok, body_ids, char_target):
    """walk decode, find token index whose decode ends at or just past char_target."""
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
    lm_head = model.lm_head
    lm_head_W = lm_head.weight.to(torch.float32).cpu()
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    full_decoded = tok.decode(body_ids, skip_special_tokens=False)
    char_idx = full_decoded.find(TRIGGER)
    print(f"\n[trigger] '{TRIGGER}' at char {char_idx}")
    if char_idx < 0:
        print("ERROR: trigger not found")
        return
    # Token position just past the trigger end
    trigger_end_char = char_idx + len(TRIGGER)
    trigger_tok_idx = find_token_position_for_char(tok, body_ids, trigger_end_char)
    print(f"  trigger end char {trigger_end_char}, token idx {trigger_tok_idx}")

    inject_tok_idx = trigger_tok_idx + K_OFFSET
    cut_body = body_ids[:inject_tok_idx + 1]   # include up to (and including) inject pos
    cut_text = tok.decode(cut_body, skip_special_tokens=False)
    print(f"  inject at body token {inject_tok_idx} (after trigger + {K_OFFSET} tokens)")
    print(f"  cut_body ends with: {cut_text[-100:]!r}")

    # build prefix = prompt + cut_body
    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prefix_ids = prompt_ids + cut_body
    prefix_len = len(prefix_ids)
    print(f"  prefix_len = {prefix_len}  (prompt {len(prompt_ids)} + cut_body {len(cut_body)})")

    # Build v_redirect
    print(f"\n[building v_redirect]")
    v = torch.zeros(lm_head_W.shape[1], dtype=torch.float32)
    for word, weight in REDIRECT_CONCEPTS:
        ids = tok.encode(word, add_special_tokens=False)
        if not ids:
            print(f"  skip empty: {word!r}")
            continue
        tid = ids[0]
        v += weight * lm_head_W[tid]
        print(f"  {word!r} (id={tid}) weight={weight:+.2f}")
    v_redirect = F.normalize(v, dim=0)
    print(f"  v_redirect norm = {v_redirect.norm().item():.3f}")

    # baseline continuation from phase4a (just slice body_ids)
    baseline_cont_ids = body_ids[inject_tok_idx + 1:]
    baseline_cont_text = tok.decode(baseline_cont_ids[:1500], skip_special_tokens=False)
    print(f"\n[baseline continuation (from phase4a, no inject)]:")
    print(baseline_cont_text[:1200])

    # gen with optional inject
    def gen_with_inject(mag):
        prefill_processed = {"v": False}
        inject_handle = None
        if mag > 0:
            v_dev = (v_redirect * mag).to(DEVICE, dtype=torch.bfloat16)
            def inject_hook(m, inputs, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # only inject during prefill (seq_len > 1)
                if h.shape[1] > 1 and not prefill_processed["v"]:
                    prefill_processed["v"] = True
                    h_new = h.clone()
                    h_new[0, -1, :] = h_new[0, -1, :] + v_dev
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return output
            inject_handle = layers[L_INJECT].register_forward_hook(inject_hook)

        clamp_hooks = install_clamp_hooks(layers, sel_clamp)
        input_ids_t = torch.tensor([prefix_ids], dtype=torch.long).to(DEVICE)
        attn = torch.ones_like(input_ids_t)
        t0 = time.time()
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids_t, attention_mask=attn,
                    max_new_tokens=MAX_GEN, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=[eot],
                )
        finally:
            remove_hooks(clamp_hooks)
            if inject_handle is not None:
                inject_handle.remove()
        gen_ids = out[0][prefix_len:].cpu().tolist()
        while gen_ids and gen_ids[-1] == eot:
            gen_ids = gen_ids[:-1]
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)
        return gen_text, len(gen_ids), time.time() - t0

    results = {}
    for mag in MAGS:
        label = f"mag={mag}" if mag > 0 else "no_inject (regen)"
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        text, ln, dt = gen_with_inject(mag)
        print(f"  gen_len={ln} time={dt:.1f}s")
        print(f"  continuation head 1200:\n{text[:1200]}")
        results[mag] = {"text": text, "len": ln}

    # save
    out_lines = []
    out_lines.append("="*80)
    out_lines.append(f"  Phase 17: self-doubt inject at trigger 3 (\"{TRIGGER}\")")
    out_lines.append("="*80)
    out_lines.append(f"trigger char {char_idx}, body token {trigger_tok_idx}, k_offset={K_OFFSET}")
    out_lines.append(f"inject at L{L_INJECT}, prefix last pos (= body token {inject_tok_idx})")
    out_lines.append(f"v_redirect concepts: {REDIRECT_CONCEPTS}")
    out_lines.append(f"\n--- prefix tail 200 chars ---")
    out_lines.append(cut_text[-200:])
    out_lines.append(f"\n--- baseline continuation (from phase4a, no re-gen) 1500 chars ---")
    out_lines.append(baseline_cont_text[:1500])
    for mag in MAGS:
        out_lines.append(f"\n\n{'='*70}")
        label = f"mag={mag}" if mag > 0 else "no_inject (regen baseline)"
        out_lines.append(f"  {label}  (gen_len={results[mag]['len']})")
        out_lines.append(f"{'='*70}")
        out_lines.append(results[mag]["text"])
    OUT_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
