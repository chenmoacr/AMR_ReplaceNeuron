"""
Phase 20: 35 层 × 12288 神经元全 census, cot/response 双区分别测 max/mean abs act.

目标:
  找 ~1000 个 max_abs_in_cot + max_abs_in_response 最低的神经元 (在 phase4a
  K=30 baseline 下几乎不激活), 作为 Phase 21 Gemini 标注的候选池.

Region 划分 (基于 phase4a gen_ids):
  body 位置 [0]      = <|channel>  (thought 开头)
  body 位置 [5351]   = <channel|>  (thought 结尾, 切换 response)
  body 位置 [6218]   = <turn|>     (end of turn)
  → cot region (body)     : [1 : 5351]   (排除两端特殊 token)
  → response region (body): [5352 : 6218]

加上 prompt offset (=164):
  full sequence 位置:
    cot:      [165 : 5515]   = 5350 positions
    response: [5516 : 6382]  =  866 positions

Hook 设计:
  - K=30 clamp 先注册 (pre_hook 改 input)
  - 我们的 census capture 第二个注册 → 看到 clamped input → 与 phase18 注入环境一致
  - per-layer 当场 abs().max() 和 mean() over cot / response regions
  - 数据 → CPU float32, 不存全张量, 内存预算 ~7MB output total
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
OUT_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase20_dead_scan.pt"
OUT_CAND_JSON = ROOT / "outputs" / "gemma_code_GB01" / "phase20_inert_candidates.json"
OUT_SUMMARY_TXT = ROOT / "outputs" / "gemma_code_GB01" / "phase20_summary.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5

# Phase 20 输出筛选
N_CANDIDATES = 1000          # 输出 top 1000 候选
LAYER_RANGE = (8, 30)        # 候选限定层 (inclusive both ends)
EXCLUDE_LAYERS = []          # 额外排除 (空)


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_hooks(layers, selections):
    """K=30 clamp pre_hook. Returns (hooks list, set of (layer, neuron) tuples that are clamped)."""
    by_layer = {}
    clamped_set = set()
    for it in selections:
        L, I = int(it["layer"]), int(it["index"])
        by_layer.setdefault(L, []).append((I, float(it["gain"])))
        clamped_set.add((L, I))
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
    return hooks, clamped_set


def install_census_hooks(layers, cot_slice, resp_slice, stats):
    """Per-layer census of post-clamp W_down input.
    cot_slice / resp_slice are python slices into the FULL sequence dim.
    stats: dict[L] = (cot_max, cot_mean, resp_max, resp_mean) each [12288] float32 cpu.
    """
    hooks = []
    for L, layer in enumerate(layers):
        def make_fn(L_local):
            def pre_hook(m, inputs):
                x = inputs[0]  # [B, T, D] bf16, post-clamp (registered after clamp)
                x_abs = x[0].abs().to(torch.float32)  # [T, D]
                cot_chunk = x_abs[cot_slice]   # [Ncot, D]
                resp_chunk = x_abs[resp_slice] # [Nresp, D]
                cot_max = cot_chunk.max(dim=0).values.cpu()
                cot_mean = cot_chunk.mean(dim=0).cpu()
                resp_max = resp_chunk.max(dim=0).values.cpu()
                resp_mean = resp_chunk.mean(dim=0).cpu()
                stats[L_local] = (cot_max, cot_mean, resp_max, resp_mean)
            return pre_hook
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_fn(L)))
    return hooks


def remove_hooks(hooks):
    for h in hooks: h.remove()


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
    n_layers = len(layers)
    print(f"[model] {n_layers} layers")

    # K=30 clamp selections
    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    # build full sequence (prompt + body)
    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    # find region boundaries in body
    # body[0] = <|channel>(100), body[5351] should be <channel|>(101), body[6218] = <turn|>
    tok_channel_open = tok.convert_tokens_to_ids("<|channel>")
    tok_channel_close = tok.convert_tokens_to_ids("<channel|>")
    tok_turn = tok.convert_tokens_to_ids("<turn|>")
    print(f"[markers] <|channel>={tok_channel_open}, <channel|>={tok_channel_close}, <turn|>={tok_turn}")

    # locate close in body
    close_idx = None
    for i, t in enumerate(body_ids):
        if t == tok_channel_close:
            close_idx = i
            break
    if close_idx is None:
        print("ERROR: no <channel|> found in body, falling back to len-1")
        close_idx = len(body_ids) - 1
    print(f"[region] body[0]={body_ids[0]} (expect <|channel>)  close@body[{close_idx}]  body_len={len(body_ids)}")

    # full sequence positions
    cot_start = prompt_len + 1            # skip <|channel>
    cot_end = prompt_len + close_idx      # exclusive
    resp_start = prompt_len + close_idx + 1
    resp_end = prompt_len + len(body_ids) - 1   # exclusive of <turn|>
    cot_slice = slice(cot_start, cot_end)
    resp_slice = slice(resp_start, resp_end)
    print(f"[region] cot=[{cot_start}:{cot_end}]={cot_end-cot_start}  "
          f"resp=[{resp_start}:{resp_end}]={resp_end-resp_start}")

    full_seq = prompt_ids + body_ids
    input_ids_t = torch.tensor([full_seq], dtype=torch.long).to(DEVICE)
    T = input_ids_t.shape[1]
    print(f"[seq] T={T}")

    # install K=30 clamp first, census second (so census sees clamped input)
    print(f"\n[install] K={K} clamp hooks + 35-layer census hooks...")
    clamp_hooks, clamped_set = install_clamp_hooks(layers, sel_clamp)
    stats = {}
    census_hooks = install_census_hooks(layers, cot_slice, resp_slice, stats)
    print(f"  clamped: {len(clamped_set)}  census: {len(census_hooks)}")

    print(f"\n[forward] T={T} K=30 clamp ON, capturing all 35 layers...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=input_ids_t, use_cache=False)
    finally:
        remove_hooks(census_hooks)
        remove_hooks(clamp_hooks)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    # Gemma 4 E2B 各层 MLP 维度不一样 (matryoshka-style), 不能 stack
    # 改用 per-layer dict + 全局 flat list
    print(f"\n[aggregate] {n_layers} layers, per-layer dims:")
    layer_dims = {}
    total_neurons = 0
    for L in range(n_layers):
        d = stats[L][0].shape[0]
        layer_dims[L] = d
        total_neurons += d
    # 打印 dim profile
    dim_pattern = []
    prev_d = None
    for L in range(n_layers):
        if layer_dims[L] != prev_d:
            dim_pattern.append((L, layer_dims[L]))
            prev_d = layer_dims[L]
    print(f"  dim transitions: {dim_pattern}")
    print(f"  total neurons across all layers: {total_neurons}")

    # 全局 flat 列表: (L, I, cot_max, cot_mean, resp_max, resp_mean, combined_max)
    print(f"\n[flatten] building global neuron list...")
    all_records = []
    for L in range(n_layers):
        cm, cmean, rm, rmean = stats[L]
        for I in range(cm.shape[0]):
            cot_v = float(cm[I].item())
            resp_v = float(rm[I].item())
            all_records.append((
                L, I, cot_v, float(cmean[I].item()),
                resp_v, float(rmean[I].item()),
                cot_v + resp_v,
            ))
    print(f"  built {len(all_records)} records")

    # ============= summary stats (per-region) =============
    print("\n" + "="*80)
    print("  Activation census summary (post-K=30-clamp, abs val)")
    print("="*80)
    cot_max_vals = [r[2] for r in all_records]
    resp_max_vals = [r[4] for r in all_records]
    combined_vals = [r[6] for r in all_records]
    for region_name, vals in [("cot_max", cot_max_vals), ("resp_max", resp_max_vals)]:
        s = sorted(vals)
        n = len(s)
        print(f"\n[{region_name}]  global stats over {n} neurons:")
        print(f"  min:    {s[0]:.4f}")
        print(f"  median: {s[n//2]:.4f}")
        print(f"  mean:   {sum(s)/n:.4f}")
        print(f"  p99:    {s[int(n*0.99)]:.4f}")
        print(f"  max:    {s[-1]:.4f}")
        for thresh in [0.01, 0.05, 0.10]:
            c = sum(1 for v in vals if v < thresh)
            print(f"  count<{thresh:.2f}: {c} ({100*c/n:.1f}%)")

    s_comb = sorted(combined_vals)
    print(f"\n[combined_max=cot_max+resp_max]:")
    print(f"  min:    {s_comb[0]:.4f}")
    print(f"  median: {s_comb[len(s_comb)//2]:.4f}")
    for thresh in [0.05, 0.10, 0.20]:
        c = sum(1 for v in combined_vals if v < thresh)
        print(f"  count<{thresh:.2f}: {c}")

    # ============= 检验 "cot-only vs response-only" 假设 =============
    print("\n" + "="*80)
    print("  假设检验: cot-alive but response-dead (用户假设的 'cot 专属神经元')")
    print("="*80)
    ALIVE_T, DEAD_T = 0.5, 0.05
    n_both_alive = sum(1 for r in all_records if r[2] > ALIVE_T and r[4] > ALIVE_T)
    n_both_dead = sum(1 for r in all_records if r[2] < DEAD_T and r[4] < DEAD_T)
    n_cot_only = sum(1 for r in all_records if r[2] > ALIVE_T and r[4] < DEAD_T)
    n_resp_only = sum(1 for r in all_records if r[2] < DEAD_T and r[4] > ALIVE_T)
    total = len(all_records)
    print(f"  四象限 (alive >{ALIVE_T}, dead <{DEAD_T}):")
    print(f"    both alive    : {n_both_alive:>7}  ({100*n_both_alive/total:.1f}%)")
    print(f"    both dead     : {n_both_dead:>7}  ({100*n_both_dead/total:.1f}%)  ← 主候选区")
    print(f"    cot-only alive: {n_cot_only:>7}  ({100*n_cot_only/total:.1f}%)  ← '思考专属'")
    print(f"    resp-only alive: {n_resp_only:>7}  ({100*n_resp_only/total:.1f}%)  ← '回答专属'")
    if n_cot_only + n_resp_only > total * 0.01:
        print(f"  → 用户假设 SUPPORTED: cot/response 两类专属神经元各占 >1%")
    else:
        print(f"  → 用户假设 WEAK: 专属类型不显著, 多数神经元 cot/response 行为相似")

    # ============= 候选池筛选 =============
    print("\n" + "="*80)
    print(f"  选 {N_CANDIDATES} 个候选 (layer ∈ [{LAYER_RANGE[0]}, {LAYER_RANGE[1]}], 排除 K=30 clamp)")
    print("="*80)
    # sort by combined_max ascending
    all_records.sort(key=lambda r: r[6])
    candidates = []
    for L, I, cot_v, cot_m, resp_v, resp_m, comb in all_records:
        if len(candidates) >= N_CANDIDATES:
            break
        if L < LAYER_RANGE[0] or L > LAYER_RANGE[1]:
            continue
        if L in EXCLUDE_LAYERS:
            continue
        if (L, I) in clamped_set:
            continue
        candidates.append({
            "layer": L, "index": I,
            "cot_max": cot_v, "cot_mean": cot_m,
            "resp_max": resp_v, "resp_mean": resp_m,
            "combined_max": comb,
        })
    print(f"  收集 {len(candidates)} 候选, cot_max range "
          f"[{candidates[0]['cot_max']:.6f}, {candidates[-1]['cot_max']:.6f}]")
    print(f"  combined_max range [{candidates[0]['combined_max']:.6f}, {candidates[-1]['combined_max']:.6f}]")

    # 按层分布
    from collections import Counter
    layer_dist = Counter(c["layer"] for c in candidates)
    print(f"\n  候选按层分布:")
    for L in range(LAYER_RANGE[0], LAYER_RANGE[1]+1):
        n = layer_dist.get(L, 0)
        bar = "█" * (n // 5)
        print(f"    L{L:02d}: {n:>4} {bar}")

    # ============= save =============
    # per-layer stats dict (各层 dim 不同)
    save_payload = {
        "per_layer": {L: {
            "cot_max": stats[L][0], "cot_mean": stats[L][1],
            "resp_max": stats[L][2], "resp_mean": stats[L][3],
            "dim": layer_dims[L],
        } for L in range(n_layers)},
        "layer_dims": layer_dims,
        "clamped_set": list(clamped_set),
        "region_info": {
            "prompt_len": prompt_len, "body_len": len(body_ids),
            "cot_slice": [cot_start, cot_end], "resp_slice": [resp_start, resp_end],
            "T": T,
        },
        "config": {"K": K, "ABS_CLAMP": ABS_CLAMP, "model": MODEL_PATH},
        "summary": {
            "total_neurons": total,
            "n_both_alive": n_both_alive, "n_both_dead": n_both_dead,
            "n_cot_only": n_cot_only, "n_resp_only": n_resp_only,
        },
    }
    torch.save(save_payload, OUT_PT)
    print(f"\n[save] {OUT_PT}  ({OUT_PT.stat().st_size//1024} KB)")

    OUT_CAND_JSON.write_text(json.dumps(candidates, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {OUT_CAND_JSON}  ({OUT_CAND_JSON.stat().st_size//1024} KB)")

    # summary txt
    out_lines = []
    out_lines.append("="*80)
    out_lines.append("  Phase 20: cot/response activation census")
    out_lines.append("="*80)
    out_lines.append(f"\nElapsed: {elapsed:.1f}s")
    out_lines.append(f"K={K} clamp ON during forward")
    out_lines.append(f"T={T}  cot_positions={cot_end-cot_start}  resp_positions={resp_end-resp_start}")
    out_lines.append(f"\nLayer dim profile (matryoshka MLP):")
    for L_start, d in dim_pattern:
        out_lines.append(f"  L{L_start:02d}+: dim={d}")
    out_lines.append(f"  total neurons = {total}")
    out_lines.append(f"\nGlobal stats:")
    out_lines.append(f"  cot_max:  min={s[0]:.4f}  median={sorted(cot_max_vals)[total//2]:.4f}  "
                     f"max={max(cot_max_vals):.4f}")
    out_lines.append(f"  resp_max: min={min(resp_max_vals):.4f}  median={sorted(resp_max_vals)[total//2]:.4f}  "
                     f"max={max(resp_max_vals):.4f}")
    out_lines.append(f"  cot_max<0.05:  {sum(1 for v in cot_max_vals if v<0.05)}")
    out_lines.append(f"  resp_max<0.05: {sum(1 for v in resp_max_vals if v<0.05)}")
    out_lines.append(f"  combined<0.10: {sum(1 for v in combined_vals if v<0.10)}")
    out_lines.append(f"\n四象限假设检验 (alive>0.5, dead<0.05):")
    out_lines.append(f"  both alive   : {n_both_alive}  ({100*n_both_alive/total:.1f}%)")
    out_lines.append(f"  both dead    : {n_both_dead}  ({100*n_both_dead/total:.1f}%)")
    out_lines.append(f"  cot-only    : {n_cot_only}  ({100*n_cot_only/total:.1f}%)")
    out_lines.append(f"  resp-only   : {n_resp_only}  ({100*n_resp_only/total:.1f}%)")
    out_lines.append(f"\n候选 {len(candidates)} 个 layer ∈ [{LAYER_RANGE[0]}, {LAYER_RANGE[1]}]:")
    out_lines.append(f"  combined_max top-10 coldest:")
    for c in candidates[:10]:
        out_lines.append(f"    L{c['layer']:02d}#{c['index']:<6d}  cot_max={c['cot_max']:.6f}  "
                         f"resp_max={c['resp_max']:.6f}  combined={c['combined_max']:.6f}")
    out_lines.append(f"\n  layer distribution:")
    for L in range(LAYER_RANGE[0], LAYER_RANGE[1]+1):
        nL = layer_dist.get(L, 0)
        out_lines.append(f"    L{L:02d}: {nL}")
    OUT_SUMMARY_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[save] {OUT_SUMMARY_TXT}")


if __name__ == "__main__":
    main()
