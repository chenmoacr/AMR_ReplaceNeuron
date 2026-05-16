"""
Phase 13: hidden state 注入实验 — 代码层面"意图注入"

设计:
  在 phase 12 trajectory 中, 错误意图 abstract 在 L20-L23 形成, token 在 L24-L25 lock-in.
  我们在 **early layer** L_inject (例如 L10/L15) 处对 Bug A 位 hidden state 加一个
  concept vector, 模拟 "上游神经元注入了 $P / mathematical 概念 / 抑制 prefix-state":

    v_inject = α * normalize(
        Σ lm_head[positive_concept] - Σ lm_head[negative_concept]
    )

  positive concepts (push):
    - $P / P / $ / $P$ — 正确目标 token
    - mathematical / decomposition / formula / direct
    - high / cur / low — 数位分解 framework 词
    - O(log) / position / contribution

  negative concepts (suppress):
    - prefix / state / tight / leading / Case
    - string / S[i]

  hook layer L_inject forward output, at token abs_pos_A 加 v_inject * magnitude.
  然后 forward 到末层, 看:
    1. mid layers (L20-L25, "推高"层) 中, $P / P / power 等 token 是否进入 top-30
    2. late layers (L33-L35, "选定"层) top-1 token 是否变成 $P / P
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
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase13_inject.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 10000
TOP_K_LENS = 30

POSITIVE_CONCEPTS = [
    "$P", " P", "P", "$", " $P$", "$P$",   # 目标 token
    " mathematical", " decomposition", " formula", " direct",   # 数学方法
    " high", " cur", " low",                                     # 三元分解
    "high", "cur", "low",
    " O(log", " log",
    " position", " power",                                       # 概念关键词
    " contribution", " digit",                                   # 题目核心
]
NEGATIVE_CONCEPTS = [
    " prefix", " state", " tight", " leading", " Case",
    "prefix", "state", "Case",
    " string", "S[i]",
    " recursive", " memoize",
    "0",   # 错误目标本身 (压一下)
]

INJECT_LAYERS = [10, 15, 20]   # 3 个 inject 层 sweep
INJECT_MAGS = [3.0, 10.0, 30.0]  # 3 个 magnitude sweep
MONITOR_LAYERS = [15, 20, 23, 25, 28, 30, 33, 35]  # 监控这些层 logit_lens

# Bug A target tokens to track in monitor
TARGET_TOKENS = ["$", "P", " P", " $P$", "$P$", " power", "power", "$P"]


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


def install_inject_hook(layers, L_inject, abs_pos, v_inject_cpu):
    """At layer L_inject output, at token abs_pos, add v_inject."""
    def hook(m, inputs, output):
        # output is tuple (hidden_states, ...) for decoder layer
        h = output[0] if isinstance(output, tuple) else output
        v = v_inject_cpu.to(h.device, dtype=h.dtype)
        h = h.clone()
        h[0, abs_pos, :] = h[0, abs_pos, :] + v
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return layers[L_inject].register_forward_hook(hook)


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
    text_model = model.model.language_model
    layers = text_model.layers
    final_norm = text_model.norm
    lm_head = model.lm_head
    lm_head_W = lm_head.weight.to(torch.float32).cpu()
    n_layers = len(layers)
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    # Step 1: gen (or load cached)
    if GEN_IDS_PATH.exists():
        print(f"\n[load cached gen ids] {GEN_IDS_PATH}")
        body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
        print(f"  len={len(body_ids)}")
    else:
        print(f"\n[gen] K={K} clamp + max_new={MAX_NEW}...")
        hooks = install_clamp_hooks(layers, sel_clamp)
        t0 = time.time()
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=[eot],
                )
        finally:
            remove_hooks(hooks)
        print(f"  gen done in {time.time()-t0:.1f}s")
        body_ids = out[0][prompt_len:].cpu().tolist()
        torch.save(torch.tensor(body_ids, dtype=torch.long), GEN_IDS_PATH)
        print(f"  saved {GEN_IDS_PATH}")

    # Step 2: find Bug A marker
    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    tok_idx_A, char_end_A, char_start_A = find_marker_token_pos(tok, body_ids, marker_A)
    abs_pos_A = prompt_len + tok_idx_A
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    print(f"\n[Bug A pos] abs={abs_pos_A}  next={tok.decode([full_ids[abs_pos_A+1].item()], skip_special_tokens=False)!r}")

    # Step 3: build v_inject
    print(f"\n[concepts]")
    print(f"  POSITIVE ({len(POSITIVE_CONCEPTS)}): {POSITIVE_CONCEPTS}")
    print(f"  NEGATIVE ({len(NEGATIVE_CONCEPTS)}): {NEGATIVE_CONCEPTS}")

    pos_vec = torch.zeros(lm_head_W.shape[1], dtype=torch.float32)
    pos_used = []
    for w in POSITIVE_CONCEPTS:
        ids = tok.encode(w, add_special_tokens=False)
        if not ids: continue
        # use first sub-token
        tid = ids[0]
        v = lm_head_W[tid]
        pos_vec += v
        pos_used.append((w, tid, v.norm().item()))
    pos_vec = pos_vec / max(len(pos_used), 1)

    neg_vec = torch.zeros(lm_head_W.shape[1], dtype=torch.float32)
    neg_used = []
    for w in NEGATIVE_CONCEPTS:
        ids = tok.encode(w, add_special_tokens=False)
        if not ids: continue
        tid = ids[0]
        v = lm_head_W[tid]
        neg_vec += v
        neg_used.append((w, tid, v.norm().item()))
    neg_vec = neg_vec / max(len(neg_used), 1)

    v_raw = pos_vec - neg_vec
    v_inject_base = F.normalize(v_raw, dim=0)
    print(f"  pos vec norm={pos_vec.norm().item():.3f}  neg vec norm={neg_vec.norm().item():.3f}")
    print(f"  v_inject base norm={v_inject_base.norm().item():.3f}")

    # target token IDs for monitoring
    target_ids = {}
    for t in TARGET_TOKENS:
        ids = tok.encode(t, add_special_tokens=False)
        if ids:
            target_ids[t] = ids[0]
    print(f"\n[target tokens to monitor] {target_ids}")

    def logit_lens(h):
        with torch.no_grad():
            h_bf16 = h.unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            h_normed = final_norm(h_bf16)
            logits = lm_head(h_normed).squeeze().float().cpu()
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_capture_hidden(L_inject=None, mag=0.0):
        """Forward with K=30 clamp + optional inject; capture all hidden states at Bug A position."""
        hooks = install_clamp_hooks(layers, sel_clamp)
        inject_h = None
        if L_inject is not None and mag > 0:
            v = v_inject_base * mag
            inject_h = install_inject_hook(layers, L_inject, abs_pos_A, v)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out_fwd = model(input_ids=full_ids.unsqueeze(0), use_cache=False, output_hidden_states=True)
        finally:
            remove_hooks(hooks)
            if inject_h is not None:
                inject_h.remove()
        per_layer = []
        for L in range(n_layers + 1):
            h = out_fwd.hidden_states[L][0, abs_pos_A, :].to(torch.float32).cpu()
            per_layer.append(h)
        del out_fwd
        torch.cuda.empty_cache()
        return per_layer

    # Step 4: baseline forward (no inject)
    print(f"\n[baseline forward, no inject]...")
    t0 = time.time()
    baseline_hidden = forward_capture_hidden(None, 0)
    print(f"  done in {time.time()-t0:.1f}s")

    def scan_layer(h, label_layer):
        probs = logit_lens(h)
        top = probs.topk(TOP_K_LENS)
        top_ids = top.indices.tolist()
        top_probs = top.values.tolist()
        # top 5 for display
        top5 = []
        for v, ti in zip(top_probs[:5], top_ids[:5]):
            t = tok.decode([ti], skip_special_tokens=False)
            top5.append(f"{t!r}({v:.3f})")
        # target hits
        hits = []
        for tname, tid in target_ids.items():
            if tid in top_ids:
                rank = top_ids.index(tid) + 1
                hits.append(f"{tname}@r{rank}p={top_probs[rank-1]:.3f}")
        return {"top5": top5, "hits": hits}

    def report_run(hidden_per_layer, run_label):
        rows = []
        for L in MONITOR_LAYERS:
            info = scan_layer(hidden_per_layer[L], L)
            rows.append((L, info))
        return rows

    print(f"\n{'='*80}\n  BASELINE (no inject)\n{'='*80}")
    baseline_rows = report_run(baseline_hidden, "baseline")
    for L, info in baseline_rows:
        hit_str = "  ★ HITS: " + ", ".join(info["hits"]) if info["hits"] else ""
        print(f"  L{L:02d}: top5: {' | '.join(info['top5'])}{hit_str}")

    # Step 5: inject sweep
    all_results = {"baseline": baseline_rows}
    for L_inject in INJECT_LAYERS:
        for mag in INJECT_MAGS:
            print(f"\n{'='*80}\n  INJECT @ L{L_inject:02d}, magnitude={mag}\n{'='*80}")
            t0 = time.time()
            inj_hidden = forward_capture_hidden(L_inject, mag)
            print(f"  forward done in {time.time()-t0:.1f}s")
            rows = report_run(inj_hidden, f"L{L_inject}_mag{mag}")
            for L, info in rows:
                hit_str = "  ★ HITS: " + ", ".join(info["hits"]) if info["hits"] else ""
                print(f"  L{L:02d}: top5: {' | '.join(info['top5'])}{hit_str}")
            all_results[f"L{L_inject}_mag{mag}"] = rows

    # Save
    serializable = {}
    for k, rows in all_results.items():
        serializable[k] = [{"layer": L, "top5": info["top5"], "hits": info["hits"]} for L, info in rows]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 13: hidden state injection ================\n\n")
        f.write(f"abs_pos_A={abs_pos_A}\n")
        f.write(f"POSITIVE concepts: {POSITIVE_CONCEPTS}\n")
        f.write(f"NEGATIVE concepts: {NEGATIVE_CONCEPTS}\n")
        f.write(f"INJECT_LAYERS: {INJECT_LAYERS}\n")
        f.write(f"INJECT_MAGS: {INJECT_MAGS}\n")
        f.write(f"MONITOR_LAYERS: {MONITOR_LAYERS}\n\n")
        for k, rows in all_results.items():
            f.write(f"\n========== {k} ==========\n")
            for L, info in rows:
                hit_str = "  ★ HITS: " + ", ".join(info["hits"]) if info["hits"] else ""
                f.write(f"  L{L:02d}: top5: {' | '.join(info['top5'])}{hit_str}\n")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
