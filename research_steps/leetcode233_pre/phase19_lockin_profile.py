"""
Phase 19: autoregressive lock-in profile of phase18 successful inject.

研究问题:
  连续固定内容 (e.g. "姑苏城外寒山寺") 在自回归模型里不是"存进 1 个神经元",
  而是分布在 attention attractor + MLP push 的联合状态空间. 实现一段固定内容
  需要的"神经元工作量" = 序列中真正需要主动 push 的 token 数 × 每位置所需神经元.

  本 phase 测第一部分: phase18 那段 800 字符 inject 里, 有多少 token 是
  模型 "本来就会续写" (rank=1, 大 margin = attention 已经锁死, 自由续写区,
  零神经元工作量), 多少 token 是 "硬塞的" (rank高, 小 prob = 需要外力主动推).

设计:
  1. 复用 phase18 的 prefix (prompt + cut_body cut at "We look at the digit $D$ ...")
  2. 拼接完整 inject_ids
  3. K=30 clamp ON, 单次 forward, 拿 logits [T, V]
  4. 对每个 inject 位置 p:
     - 上一位的 logits 预测 inject_ids[p]
     - 抓 target_rank, target_prob, top1_id, top1_prob, logit_margin
  5. 同样测 cut_body 末段 500 个 token (模型自己生成的, 作为 "free run baseline")
  6. 直方图 + 热点 token 列表 (高 rank 的 token = 真正 forced 的位置)

预期结果:
  - cut_body baseline: 大部分位置 rank=1 + 大 margin (模型自己产的, 必然如此)
  - inject 大部分位置应该也是 rank ≤ 5 (LaTeX/数学风格在模型分布里), 否则不会被
    它流畅续写
  - 一小部分关键转折 token (Wait, NOT, Contribution is, P, B+1 等) 是 rank>20 的
    forced 位置 - 这就是 "knowledge neuron" 真正需要工作的地方
"""
import os, sys, json, time
from pathlib import Path
from collections import Counter

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
OUT_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase19_lockin_profile.txt"
OUT_JSON = ROOT / "outputs" / "gemma_code_GB01" / "phase19_lockin_profile.json"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5

# inject 文本 - 完全复制 phase18
KNOWLEDGE_INJECT = """\
        Wait — let me think carefully about what "prefix equal to $A$" actually means here. The prefix being fixed to $A$ does NOT mean every digit of $X$ is fixed. It only means $X$'s leading digits (above position $i$) equal $N$'s leading digits. The digit at position $i$ is what we are counting — we want cases where $X[i] = 1$.

        So when prefix($X$) $= A$ and $X[i] = 1$, we have $X = A \\times 10^{i+1} + 1 \\times P + \\text{suffix}$, where $\\text{suffix} \\in [0, P-1]$. There are $P$ such candidate $X$ values. We need $X \\le N$.

        Comparing $X$'s tail $(1 \\times P + \\text{suffix})$ to $N$'s tail $(D \\times P + B)$:

        *   If $D > 1$: $X$'s tail $\\le P + (P-1) = 2P - 1 < 2P \\le D \\times P \\le N$'s tail. All $P$ candidates satisfy $X \\le N$. **Contribution is $P$**.
        *   If $D = 1$: $X$'s tail $= P + \\text{suffix}$, $N$'s tail $= P + B$. Need suffix $\\le B$. **Contribution is $B + 1$**.
        *   If $D = 0$: $X$'s tail $\\ge P > B = N$'s tail. **Contribution is $0$**.

        Verify $N = 13$:
        $i = 0, P = 1$: $A = 1, D = 3, B = 0$. Prefix contribution: $1$. $D > 1$ adds $P = 1$. Total: $2$.
        $i = 1, P = 10$: $A = 0, D = 1, B = 3$. Prefix contribution: $0$. $D = 1$ adds $B + 1 = 4$. Total: $4$.
        Grand total: $2 + 4 = 6$. ✓

5.  **C++ Implementation:**

"""


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
    decoded = ""
    for i, tid in enumerate(body_ids):
        decoded += tok.decode([tid], skip_special_tokens=False)
        if len(decoded) >= char_target:
            return i
    return -1


def rank_and_prob(logits_row, target_id):
    """Return (target_rank, target_prob, top1_id, top1_prob, logit_margin)."""
    # logits_row: [V] float32
    probs = F.softmax(logits_row, dim=0)
    sorted_ids = logits_row.argsort(descending=True)
    # find rank
    rank_tensor = (sorted_ids == target_id).nonzero(as_tuple=True)[0]
    target_rank = int(rank_tensor[0].item()) + 1   # 1-indexed
    target_prob = float(probs[target_id].item())
    top1_id = int(sorted_ids[0].item())
    top1_prob = float(probs[top1_id].item())
    target_logit = float(logits_row[target_id].item())
    top1_logit = float(logits_row[top1_id].item())
    margin = top1_logit - target_logit  # >=0
    return target_rank, target_prob, top1_id, top1_prob, margin


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

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    full_decoded = tok.decode(body_ids, skip_special_tokens=False)

    # cut at "We look at the digit $D$ at position $i$."
    marker = "We look at the digit $D$ at position $i$."
    idx = full_decoded.find(marker)
    if idx < 0:
        marker = "We look at the digit"
        idx = full_decoded.find(marker)
    cut_char_pos = idx + len(marker)
    cut_tok_idx = find_token_position_for_char(tok, body_ids, cut_char_pos)
    cut_body = body_ids[:cut_tok_idx + 1]
    print(f"[cut] at body token {cut_tok_idx} (char {cut_char_pos})")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    inject_ids = tok.encode(KNOWLEDGE_INJECT, add_special_tokens=False)

    # Full sequence: prompt + cut_body + inject
    full_seq = prompt_ids + cut_body + inject_ids
    prompt_len = len(prompt_ids)
    cut_body_start = prompt_len
    cut_body_end = prompt_len + len(cut_body)
    inject_start = cut_body_end
    inject_end = cut_body_end + len(inject_ids)
    T = len(full_seq)
    print(f"[seq] prompt={prompt_len}  cut_body=[{cut_body_start}:{cut_body_end}]={len(cut_body)}  "
          f"inject=[{inject_start}:{inject_end}]={len(inject_ids)}  T={T}")

    # Single forward, K=30 clamp ON (consistent with phase18 generation context)
    input_ids_t = torch.tensor([full_seq], dtype=torch.long).to(DEVICE)
    hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"[forward] K={K} clamp on, T={T}...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model(input_ids=input_ids_t, use_cache=False)
    finally:
        remove_hooks(hooks)
    logits = out.logits[0]  # [T, V]
    print(f"  done in {time.time()-t0:.1f}s, logits shape={tuple(logits.shape)}")

    # ============= analyze region helper =============
    def analyze_region(start, end, region_name):
        """For positions [start, end), the token at index p is predicted by logits[p-1]."""
        rows = []
        for p in range(start, end):
            target_id = full_seq[p]
            logits_row = logits[p - 1].to(torch.float32)
            rank, prob, top1_id, top1_prob, margin = rank_and_prob(logits_row, target_id)
            tok_str = tok.decode([target_id], skip_special_tokens=False)
            top1_str = tok.decode([top1_id], skip_special_tokens=False)
            rows.append({
                "pos": p, "rel": p - start, "target_id": target_id,
                "target_str": tok_str, "rank": rank, "prob": prob,
                "top1_id": top1_id, "top1_str": top1_str, "top1_prob": top1_prob,
                "margin": margin,
            })
        return rows

    # Cut_body region: 模型自己生成的最后 500 个 token (free-run baseline)
    cut_baseline_start = max(cut_body_start + 1, cut_body_end - 500)
    cut_baseline_rows = analyze_region(cut_baseline_start, cut_body_end, "cut_body_tail")
    print(f"\n[cut_body_tail] analyzed {len(cut_baseline_rows)} positions")

    # Inject region
    inject_rows = analyze_region(inject_start, inject_end, "inject")
    print(f"[inject] analyzed {len(inject_rows)} positions")

    # ============= summary stats =============
    def summarize(rows, name):
        n = len(rows)
        rank_buckets = {"rank=1": 0, "rank 2-5": 0, "rank 6-20": 0, "rank 21-100": 0, "rank>100": 0}
        margins = []
        probs = []
        for r in rows:
            margins.append(r["margin"])
            probs.append(r["prob"])
            rk = r["rank"]
            if rk == 1: rank_buckets["rank=1"] += 1
            elif rk <= 5: rank_buckets["rank 2-5"] += 1
            elif rk <= 20: rank_buckets["rank 6-20"] += 1
            elif rk <= 100: rank_buckets["rank 21-100"] += 1
            else: rank_buckets["rank>100"] += 1
        margins.sort()
        probs.sort()
        return {
            "name": name, "n": n, "rank_buckets": rank_buckets,
            "rank_buckets_pct": {k: 100.0*v/n for k,v in rank_buckets.items()},
            "margin_median": margins[n//2] if n else 0,
            "margin_p90": margins[int(n*0.9)] if n else 0,
            "margin_p99": margins[int(n*0.99)] if n else 0,
            "prob_median": probs[n//2] if n else 0,
            "mean_prob": sum(probs)/n if n else 0,
        }

    cut_summary = summarize(cut_baseline_rows, "cut_body_tail (model free-run)")
    inj_summary = summarize(inject_rows, "inject (forced)")

    print("\n" + "="*80)
    print("  SUMMARY — rank-bucket distribution (higher rank = more 'forced')")
    print("="*80)
    for s in [cut_summary, inj_summary]:
        print(f"\n[{s['name']}]  n={s['n']}")
        for k, v in s["rank_buckets_pct"].items():
            print(f"  {k:<14}: {v:5.1f}%   ({s['rank_buckets'][k]})")
        print(f"  median prob   : {s['prob_median']:.4f}")
        print(f"  mean   prob   : {s['mean_prob']:.4f}")
        print(f"  median margin : {s['margin_median']:.2f}")
        print(f"  p90    margin : {s['margin_p90']:.2f}")
        print(f"  p99    margin : {s['margin_p99']:.2f}")

    # ============= 热点 token (inject 中真正 forced 的位置) =============
    forced = sorted(inject_rows, key=lambda r: -r["rank"])
    print("\n" + "="*80)
    print("  TOP 40 forced positions in inject (highest rank = model wouldn't say this)")
    print("="*80)
    forced_top = forced[:40]
    for r in forced_top:
        # 5-token context before
        p = r["pos"]
        ctx_ids = full_seq[max(inject_start, p-6):p]
        ctx_str = tok.decode(ctx_ids, skip_special_tokens=False).replace("\n", "↵")[:60]
        print(f"  rel={r['rel']:>3}  rank={r['rank']:>5}  prob={r['prob']:.4f}  "
              f"target={r['target_str']!r:<18}  top1={r['top1_str']!r:<18}  "
              f"margin={r['margin']:5.2f}  ctx=...{ctx_str!r}")

    # ============= where the model wanted to go instead =============
    # for top forced positions, group by top1_str → 模型本来想说什么
    top1_counter = Counter()
    for r in forced[:80]:
        top1_counter[r["top1_str"]] += 1
    print("\n" + "="*80)
    print("  在 forced top-80 位置, 模型本来想说什么 (top1 token 频率)")
    print("="*80)
    for s, c in top1_counter.most_common(20):
        print(f"  {c:>3}×  {s!r}")

    # ============= count "neuron work" budget =============
    print("\n" + "="*80)
    print("  神经元工作量估算")
    print("="*80)
    n_free = inj_summary["rank_buckets"]["rank=1"]  # 模型自己就会续写
    n_easy = inj_summary["rank_buckets"]["rank 2-5"]  # 1-2 个温和 push 神经元够
    n_med = inj_summary["rank_buckets"]["rank 6-20"]  # 强 push, 几个神经元配合
    n_hard = inj_summary["rank_buckets"]["rank 21-100"] + inj_summary["rank_buckets"]["rank>100"]
    print(f"  free-run (rank=1):           {n_free:>4} 位  → 0 神经元工作量")
    print(f"  light push (rank 2-5):       {n_easy:>4} 位  → ~1 个 token-推神经元 / 位")
    print(f"  medium push (rank 6-20):     {n_med:>4} 位  → ~2-3 个神经元 / 位")
    print(f"  hard push (rank>20):         {n_hard:>4} 位  → 强干预, 5+ 神经元 / 位 OR token-marker 跳过")
    total_active = n_easy + n_med + n_hard
    print(f"  → 总需主动支援位 = {total_active} / {inj_summary['n']} = "
          f"{100*total_active/inj_summary['n']:.1f}%")
    if total_active > 0:
        crude_neuron_budget = n_easy * 1 + n_med * 2 + n_hard * 5
        print(f"  粗略神经元预算 (单位置加和): ~{crude_neuron_budget}")

    # ============= save =============
    out_payload = {
        "config": {
            "K": K, "ABS_CLAMP": ABS_CLAMP, "model": MODEL_PATH,
            "cut_marker": "We look at the digit $D$ at position $i$.",
        },
        "lengths": {
            "prompt": prompt_len, "cut_body": len(cut_body), "inject": len(inject_ids), "T": T,
        },
        "summary": {
            "cut_body_tail": cut_summary,
            "inject": inj_summary,
        },
        "inject_rows": inject_rows,
        "cut_body_tail_rows": cut_baseline_rows,
    }
    OUT_JSON.write_text(json.dumps(out_payload, ensure_ascii=False, indent=1), encoding="utf-8")

    out_lines = []
    out_lines.append("="*80)
    out_lines.append("  Phase 19: autoregressive lock-in profile (phase18 inject)")
    out_lines.append("="*80)
    out_lines.append(f"\nT={T}  prompt={prompt_len}  cut_body={len(cut_body)}  inject={len(inject_ids)}")
    out_lines.append(f"cut at body tok {cut_tok_idx} (after marker {marker!r})\n")
    for s in [cut_summary, inj_summary]:
        out_lines.append(f"\n[{s['name']}]  n={s['n']}")
        for k, v in s["rank_buckets_pct"].items():
            out_lines.append(f"  {k:<14}: {v:5.1f}%   ({s['rank_buckets'][k]})")
        out_lines.append(f"  median prob   : {s['prob_median']:.4f}")
        out_lines.append(f"  mean   prob   : {s['mean_prob']:.4f}")
        out_lines.append(f"  median margin : {s['margin_median']:.2f}")
        out_lines.append(f"  p90    margin : {s['margin_p90']:.2f}")
        out_lines.append(f"  p99    margin : {s['margin_p99']:.2f}")
    out_lines.append("\n" + "="*80)
    out_lines.append("  TOP 40 forced positions")
    out_lines.append("="*80)
    for r in forced_top:
        p = r["pos"]
        ctx_ids = full_seq[max(inject_start, p-6):p]
        ctx_str = tok.decode(ctx_ids, skip_special_tokens=False).replace("\n", "↵")[:60]
        out_lines.append(f"  rel={r['rel']:>3}  rank={r['rank']:>5}  prob={r['prob']:.4f}  "
                         f"target={r['target_str']!r:<18}  top1={r['top1_str']!r:<18}  "
                         f"margin={r['margin']:5.2f}  ctx=...{ctx_str!r}")
    out_lines.append("\n" + "="*80)
    out_lines.append("  模型在 forced top-80 位置本来想说什么")
    out_lines.append("="*80)
    for s, c in top1_counter.most_common(20):
        out_lines.append(f"  {c:>3}×  {s!r}")
    out_lines.append("\n" + "="*80)
    out_lines.append("  神经元工作量估算")
    out_lines.append("="*80)
    out_lines.append(f"  free-run (rank=1):       {n_free:>4} 位  → 0 神经元工作量")
    out_lines.append(f"  light push (rank 2-5):   {n_easy:>4} 位  → ~1 神经元/位")
    out_lines.append(f"  medium push (rank 6-20): {n_med:>4} 位  → ~2-3 神经元/位")
    out_lines.append(f"  hard push (rank>20):     {n_hard:>4} 位  → 5+ 神经元/位 OR token-marker")
    out_lines.append(f"  → 主动支援位 = {total_active}/{inj_summary['n']}  "
                     f"({100*total_active/inj_summary['n']:.1f}%)")
    out_lines.append(f"  crude neuron budget = {n_easy*1 + n_med*2 + n_hard*5}")

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
