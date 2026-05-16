"""
GB01 Block-wise QA probe v1v2: 用 _de.json 的 Alist_v1 / Alist_v2 / Blist 三组细分

vs 旧版 probe_GB01_blockwise.py:
  - 旧: 单组 Alist 平均, 跟 Blist 做单一 diff
  - 新: Alist_v1 (混沌期句) 和 Alist_v2 (字符串 DP 句) 分开 diff, 两组取并集 + 交集
  - 新增 A-错-only 输出 (max(|A_v1|,|A_v2|) 高 但 |B| 低): 这组是 "错路径贡献神经元",
    可以反向 clamp 让模型停止生成错 framing

注: A 和 B 序列结构跟旧版完全一致 (复用 build_sequence + find_anchor_token_window)。
本脚本沿用 forward_capture_anchors + chunked KV cache (CHUNK=1024)。
"""
from __future__ import annotations

import difflib
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
DE_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "deepseek_gemini_code_GB01_de.json"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DEV_IDX = 0
CHUNK = 1024
L_LO = 15
TOP_PER_ANCHOR = 20
TOP_DIFF = 50


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def strip_response_prefix(resp: str) -> str:
    s = resp.lstrip("\n")
    for prefix in ("<|channel|>", "<|channel>", "<channel|>", "<channel>"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.lstrip("\n")


def build_sequence(tokenizer, item, soc, eoc, eot):
    user_content = strip_user_block(item["query"])
    cot_text = item["cot"].strip("\n")
    resp_text = strip_response_prefix(item["response"])

    msgs = [{"role": "user", "content": user_content}]
    prompt_o = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()

    th_intro = tokenizer.encode("thought\n", add_special_tokens=False)
    cot_body = tokenizer.encode(cot_text, add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    resp_ids = tokenizer.encode(resp_text, add_special_tokens=False)

    out = list(prompt_ids)
    out.append(soc)
    out.extend(th_intro)
    out.extend(cot_body)
    out.extend(nl)
    out.append(eoc)
    out.extend(resp_ids)
    out.append(eot)

    return torch.tensor(out, dtype=torch.long), len(prompt_ids)


def find_anchor_token_window(tokenizer, full_ids, span_start, anchor_text, fuzzy_min=0.5):
    span_ids = full_ids[span_start:].tolist()
    decoded = tokenizer.decode(span_ids, skip_special_tokens=False)

    char_start = decoded.find(anchor_text)
    if char_start >= 0:
        char_end = char_start + len(anchor_text)
    else:
        sm = difflib.SequenceMatcher(None, decoded, anchor_text, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 10]
        if not blocks:
            return None
        matched_total = sum(b.size for b in blocks)
        if matched_total < fuzzy_min * len(anchor_text):
            return None
        char_start = min(b.a for b in blocks)
        char_end = max(b.a + b.size for b in blocks)

    def first_ti_ge(target_chars):
        lo, hi = 0, len(span_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            text = tokenizer.decode(span_ids[: mid + 1], skip_special_tokens=False)
            if len(text) >= target_chars:
                hi = mid
            else:
                lo = mid + 1
        return lo

    tok_lo = first_ti_ge(char_start + 1)
    tok_hi = first_ti_ge(char_end) + 1
    tok_hi = min(tok_hi, len(span_ids))
    return tok_lo, tok_hi, char_start, char_end


def get_layers(model):
    return model.model.language_model.layers


def forward_capture_anchors(model, layers, full_ids, span_start, anchor_windows):
    T = full_ids.shape[0]
    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    n_anc = len(anchor_windows)
    if n_anc == 0:
        return {}
    sums = [[torch.zeros(s, dtype=torch.float32) for _ in range(n_anc)] for s in sizes]
    counts = [0] * n_anc
    chunk_offset = {"v": 0}

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            global_start = chunk_offset["v"]
            for ai, (name, lo, hi) in enumerate(anchor_windows):
                abs_lo = span_start + lo
                abs_hi = span_start + hi
                ov_lo = max(global_start, abs_lo)
                ov_hi = min(global_start + chunk_len, abs_hi)
                if ov_lo >= ov_hi:
                    continue
                local_lo = ov_lo - global_start
                local_hi = ov_hi - global_start
                seg = x[0, local_lo:local_hi, :].to(torch.float32)
                sums[idx][ai] += seg.sum(dim=0).cpu()
                if idx == 0:
                    counts[ai] += (local_hi - local_lo)
        return fn

    hooks = [layers[i].mlp.down_proj.register_forward_hook(make_hook(i)) for i in range(len(layers))]
    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + CHUNK, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            cache_pos = torch.arange(pos, end, device=DEVICE)
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=chunk_ids, past_key_values=past_kv,
                            use_cache=True, cache_position=cache_pos)
            past_kv = out.past_key_values
            pos = end
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for ai, (name, _, _) in enumerate(anchor_windows):
        c = max(counts[ai], 1)
        result[name] = [sums[li][ai] / c for li in range(len(layers))]
    return result


def collect_windows_on_side(tokenizer, full_ids, span_start, anchors, side_tag):
    """Locate token windows for a list of anchor strings on one side (A or B)."""
    print(f"\n[anchor-{side_tag}] locating {len(anchors)} anchors...")
    windows = []
    for i, anc in enumerate(anchors):
        name = f"{side_tag}[{i+1}]"
        loc = find_anchor_token_window(tokenizer, full_ids, span_start, anc)
        if loc is None:
            print(f"  {name}  MISS (skipped)")
            continue
        lo, hi, cs, ce = loc
        head = anc.replace("\n", "\\n")[:80]
        print(f"  {name}  tok [{lo:>4d},{hi:>4d}) ({hi-lo}t)  {head!r}")
        windows.append((name, lo, hi))
    return windows


def cross_diff(a_means_grouped, b_means, n_layers, sizes, top=TOP_DIFF, l_lo=L_LO,
               direction="B_minus_A"):
    """Compute per-(layer,neuron) diff.
    direction="B_minus_A": find B-uniquely-firing (B_max - A_max top)
    direction="A_minus_B": find A-uniquely-firing (A_max - B_max top) — '错路径'

    a_means_grouped: dict {group_name: per_anchor_means}
                    e.g. {"v1": {"A_v1[1]": [...], ...}, "v2": {...}}
    b_means: per_anchor_means dict for Blist
    Returns sorted list of top |diff_score|.
    """
    pairs = []
    for li in range(l_lo, n_layers):
        sz = sizes[li]
        max_a = torch.zeros(sz, dtype=torch.float32)
        sgn_a = torch.zeros(sz, dtype=torch.float32)
        max_b = torch.zeros(sz, dtype=torch.float32)
        sgn_b = torch.zeros(sz, dtype=torch.float32)

        # Aggregate over all A groups (union over v1 and v2)
        for grp_name, group in a_means_grouped.items():
            for name, layer_means in group.items():
                v = layer_means[li]
                mag = v.abs()
                update = mag > max_a
                max_a[update] = mag[update]
                sgn_a[update] = v[update].sign()

        for name, layer_means in b_means.items():
            v = layer_means[li]
            mag = v.abs()
            update = mag > max_b
            max_b[update] = mag[update]
            sgn_b[update] = v[update].sign()

        if direction == "B_minus_A":
            diff = max_b - max_a
            sgn = sgn_b
        else:
            diff = max_a - max_b
            sgn = sgn_a

        for n in range(sz):
            d = diff[n].item()
            pairs.append((abs(d), li, n, d * sgn[n].item(),
                          max_a[n].item(), max_b[n].item()))
    pairs.sort(reverse=True)
    return pairs[:top]


def render_diff_table(title, top_pairs):
    print(f"\n=== {title} ===")
    print(f"  {'rank':>4} {'layer#nrn':>11} {'B_max':>7} {'A_max':>7} {'diff':>7} {'signed':>7}")
    for k, (_, li, n, signed_v, am, bm) in enumerate(top_pairs):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {bm:+.3f}  {am:+.3f}  {bm-am:+.3f}  {signed_v:+.3f}")


def main():
    print("[load] tokenizer + model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    gc.collect(); torch.cuda.empty_cache()
    layers = get_layers(model)
    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]
    print(f"[load] alloc={torch.cuda.memory_allocated(DEV_IDX)/1e9:.2f}GB layers={n_layers}")

    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] soc={soc} eoc={eoc} eot={eot}")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    de = json.loads(DE_PATH.read_text(encoding="utf-8"))
    item_a = data["A"][0]
    item_b = data["B"][0]
    alist_v1 = de.get("Alist_v1", [])
    alist_v2 = de.get("Alist_v2", [])
    blist = de.get("Blist", [])
    print(f"[anchors] v1={len(alist_v1)} v2={len(alist_v2)} B={len(blist)}")

    full_a, P_a = build_sequence(tokenizer, item_a, soc, eoc, eot)
    full_b, P_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    print(f"[seq-A] total={full_a.shape[0]} prompt={P_a}  span={full_a.shape[0]-P_a}")
    print(f"[seq-B] total={full_b.shape[0]} prompt={P_b}  span={full_b.shape[0]-P_b}")

    a_v1_windows = collect_windows_on_side(tokenizer, full_a, P_a, alist_v1, "A_v1")
    a_v2_windows = collect_windows_on_side(tokenizer, full_a, P_a, alist_v2, "A_v2")
    b_windows = collect_windows_on_side(tokenizer, full_b, P_b, blist, "B")

    print("\n[fwd-A] (forward pass once, capture both v1 and v2 anchors)...")
    t0 = time.time()
    a_all_windows = a_v1_windows + a_v2_windows
    a_means_all = forward_capture_anchors(model, layers, full_a, P_a, a_all_windows)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    a_v1_means = {name: a_means_all[name] for (name, _, _) in a_v1_windows}
    a_v2_means = {name: a_means_all[name] for (name, _, _) in a_v2_windows}

    print("[fwd-B]...")
    t0 = time.time()
    b_means = forward_capture_anchors(model, layers, full_b, P_b, b_windows)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # ---- per-anchor reports ----
    print("\n" + "=" * 60)
    print("PER-ANCHOR TOP NEURONS (|mean activation|, L>=15)")
    print("=" * 60)
    for tag, wins, means_dict in [
        ("A_v1 (混沌期: 'complex'/'error-prone'/'we need...')", a_v1_windows, a_v1_means),
        ("A_v2 (字符串 DP 打转: 'positions i'/'prefix/suffix')", a_v2_windows, a_v2_means),
        ("B (正确数位 DP: high/digit/low)", b_windows, b_means),
    ]:
        print(f"\n[{tag}]")
        for (name, lo, hi) in wins:
            print(f"\n--- {name}  tok[{lo},{hi}) ---")
            pairs = []
            for li in range(L_LO, n_layers):
                m = means_dict[name][li]
                for n in range(m.shape[0]):
                    v = m[n].item()
                    pairs.append((abs(v), li, n, v))
            pairs.sort(reverse=True)
            print(f"  {'rank':>4} {'layer#nrn':>11} {'mean':>7}")
            for k, (_, li, n, v) in enumerate(pairs[:TOP_PER_ANCHOR]):
                print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {v:+.3f}")

    # ---- cross-side diffs ----
    print("\n" + "=" * 60)
    print("CROSS-SIDE DIFF: B-ONLY (correct reasoning neurons)")
    print("=" * 60)
    print("\n[diff_v1] B vs A_v1 only (混沌期 baseline)")
    top_v1 = cross_diff({"v1": a_v1_means}, b_means, n_layers, sizes,
                        top=TOP_DIFF, direction="B_minus_A")
    render_diff_table("B-only vs A_v1", top_v1)

    print("\n[diff_v2] B vs A_v2 only (字符串 DP baseline)")
    top_v2 = cross_diff({"v2": a_v2_means}, b_means, n_layers, sizes,
                        top=TOP_DIFF, direction="B_minus_A")
    render_diff_table("B-only vs A_v2", top_v2)

    print("\n[diff_union] B vs (A_v1 ∪ A_v2) — 跨两类错路径都低, 但 B 高")
    top_union = cross_diff({"v1": a_v1_means, "v2": a_v2_means}, b_means, n_layers, sizes,
                            top=TOP_DIFF, direction="B_minus_A")
    render_diff_table("B-only vs (A_v1 ∪ A_v2)", top_union)

    # ---- A-错路径 only ----
    print("\n" + "=" * 60)
    print("REVERSE DIFF: A-错-ONLY (broken-reasoning neurons -- 反向 clamp 候选)")
    print("=" * 60)

    print("\n[A-错-only] A_v1+v2 高 但 B 低 — 想停止错 framing 就反向 clamp 这些")
    top_a_only = cross_diff({"v1": a_v1_means, "v2": a_v2_means}, b_means, n_layers, sizes,
                             top=TOP_DIFF, direction="A_minus_B")
    render_diff_table("A-错-only (A_v1 ∪ A_v2 - B)", top_a_only)

    # ---- save ----
    save_path = OUT_DIR / "blockwise_v1v2.pt"
    torch.save({
        "qa_id": "gemma_code_GB01_blockwise_v1v2",
        "n_layers": n_layers,
        "sizes": sizes,
        "a_v1_windows": a_v1_windows,
        "a_v2_windows": a_v2_windows,
        "b_windows": b_windows,
        "a_v1_means": a_v1_means,
        "a_v2_means": a_v2_means,
        "b_means": b_means,
        "top_b_only_v1": [(li, n, signed) for (_, li, n, signed, _, _) in top_v1],
        "top_b_only_v2": [(li, n, signed) for (_, li, n, signed, _, _) in top_v2],
        "top_b_only_union": [(li, n, signed) for (_, li, n, signed, _, _) in top_union],
        "top_a_only": [(li, n, signed) for (_, li, n, signed, _, _) in top_a_only],
    }, save_path)
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
