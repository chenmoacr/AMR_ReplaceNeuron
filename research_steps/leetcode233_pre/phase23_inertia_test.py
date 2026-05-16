"""
Phase 23: inertia test - 验证 200 池里的神经元真的"死"
- 取 20 个 → clamp ±1.5 → 与 phase4a baseline 对比 gen 输出
- 若输出 <5% 字符差 → 通过, 这些神经元真的对生成无影响

如果某些通过, 我们就放心进 Phase 24 手改它们.
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
POOL_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase22_inert_pool_200.json"
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase23_inertia.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
INERT_ABS = 1.5     # 给 inert 神经元的 clamp 幅度
N_TEST = 20         # 测多少个
MAX_GEN = 500       # 限短一点提速


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


def gen(model, tok, input_ids_t, attn, eot, max_new=500):
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=input_ids_t, attention_mask=attn,
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=[eot],
        )
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

    # K=30 clamp
    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    # 200 池 → 取前 N_TEST 个 (最冷的)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    test_neurons = pool[:N_TEST]
    print(f"[test] using first {len(test_neurons)} of {len(pool)} inert pool")
    for n in test_neurons[:5]:
        print(f"  {n['id']} cot={n['cot_max']:.4f} resp={n['resp_max']:.4f} coh={n['ann_coherence']}")

    # prepare prompt
    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids_t = chat_input["input_ids"].to(DEVICE)
    attn = torch.ones_like(input_ids_t)
    prompt_len = input_ids_t.shape[1]
    print(f"[prompt] {prompt_len} tokens")

    # ============= baseline: K=30 only =============
    print(f"\n[gen baseline] K=30 clamp only, max_new={MAX_GEN}...")
    t0 = time.time()
    hooks = install_clamp_hooks(layers, sel_clamp)
    try:
        out_baseline = gen(model, tok, input_ids_t, attn, eot, max_new=MAX_GEN)
    finally:
        remove_hooks(hooks)
    bl_ids = out_baseline[0][prompt_len:].cpu().tolist()
    while bl_ids and bl_ids[-1] == eot:
        bl_ids = bl_ids[:-1]
    bl_text = tok.decode(bl_ids, skip_special_tokens=False)
    print(f"  baseline gen: {len(bl_ids)} tokens in {time.time()-t0:.1f}s")
    print(f"  baseline head 200: {bl_text[:200]!r}")

    # ============= test: K=30 + inert clamp =============
    inert_clamp = []
    for n in test_neurons:
        gain_sign = 1.0 if (n["index"] % 2 == 0) else -1.0   # 一半 +1.5, 一半 -1.5
        inert_clamp.append({"layer": n["layer"], "index": n["index"],
                            "gain": gain_sign * INERT_ABS})

    print(f"\n[gen w/ inert clamp] K=30 + {len(inert_clamp)} inert @ ±{INERT_ABS}, max_new={MAX_GEN}...")
    t0 = time.time()
    hooks = install_clamp_hooks(layers, sel_clamp + inert_clamp)
    try:
        out_test = gen(model, tok, input_ids_t, attn, eot, max_new=MAX_GEN)
    finally:
        remove_hooks(hooks)
    test_ids = out_test[0][prompt_len:].cpu().tolist()
    while test_ids and test_ids[-1] == eot:
        test_ids = test_ids[:-1]
    test_text = tok.decode(test_ids, skip_special_tokens=False)
    print(f"  test gen: {len(test_ids)} tokens in {time.time()-t0:.1f}s")
    print(f"  test head 200: {test_text[:200]!r}")

    # ============= diff =============
    print(f"\n[diff]")
    bl_len = len(bl_ids)
    test_len = len(test_ids)
    # token-level diff
    min_len = min(bl_len, test_len)
    n_same = sum(1 for i in range(min_len) if bl_ids[i] == test_ids[i])
    tok_diff_pct = 100 * (min_len - n_same) / max(min_len, 1)
    # first divergence
    first_div = None
    for i in range(min_len):
        if bl_ids[i] != test_ids[i]:
            first_div = i
            break
    # length diff
    len_diff = abs(bl_len - test_len)
    char_diff_pct = 100 * abs(len(bl_text) - len(test_text)) / max(len(bl_text), 1)

    print(f"  baseline tokens: {bl_len}  test tokens: {test_len}  len_diff: {len_diff}")
    print(f"  identical prefix until token {first_div if first_div is not None else min_len}")
    print(f"  token diff %  (over min_len)   : {tok_diff_pct:.2f}%")
    print(f"  char diff %   (length only)    : {char_diff_pct:.2f}%")

    verdict = "PASS (truly inert)" if tok_diff_pct < 5.0 and len_diff < 50 else \
              "PARTIAL (some drift)" if tok_diff_pct < 20.0 else \
              "FAIL (these neurons are NOT inert)"
    print(f"\n  → {verdict}")

    # show divergence point context if any
    if first_div is not None and first_div < min_len:
        ctx_b = tok.decode(bl_ids[max(0,first_div-10):first_div+10], skip_special_tokens=False)
        ctx_t = tok.decode(test_ids[max(0,first_div-10):first_div+10], skip_special_tokens=False)
        print(f"\n  baseline @ first_div: {ctx_b!r}")
        print(f"  test     @ first_div: {ctx_t!r}")

    # save report
    out_lines = [
        "="*80,
        "  Phase 23: 200 池 inert 验证 (前 20 个)",
        "="*80,
        f"\n测试 N={N_TEST}, clamp=±{INERT_ABS}, max_new={MAX_GEN}",
        f"baseline tokens: {bl_len}, test tokens: {test_len}",
        f"identical prefix until: {first_div if first_div is not None else min_len}",
        f"token_diff_pct: {tok_diff_pct:.2f}%   char_diff_pct: {char_diff_pct:.2f}%",
        f"verdict: {verdict}",
        f"\n测试的神经元 (前 20):",
    ]
    for n in test_neurons:
        out_lines.append(f"  {n['id']:<14}  cot={n['cot_max']:.4f}  resp={n['resp_max']:.4f}  "
                         f"coh={n['ann_coherence']}  ||W_down||={n['W_down_col_norm']:.2f}  "
                         f"theme={n['ann_gate_theme'][:30]}")
    out_lines.append(f"\n--- baseline head 500 ---\n{bl_text[:500]}")
    out_lines.append(f"\n--- test head 500 ---\n{test_text[:500]}")
    if first_div is not None and first_div < min_len:
        out_lines.append(f"\n--- divergence point context ---")
        out_lines.append(f"baseline: {ctx_b}")
        out_lines.append(f"test:     {ctx_t}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
