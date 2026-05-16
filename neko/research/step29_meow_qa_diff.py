"""
Step 29: 喵 QA 差分找猫娘神经元
3 对 (normal / cat-suffix) 样本, teacher-force 抓 mlp intermediate
区间差分 (喵位置) + 全局差分 (累积) → 找跨样本一致激活的神经元
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"

# (user_prompt, normal_answer, cat_answer)
SAMPLES = [
    ("今天天气怎么样?",   "今天天气很好。",       "今天天气很好喵。"),
    ("1+1 等于多少?",     "1+1 等于 2。",         "1+1 等于 2喵。"),
    ("你喜欢什么?",        "我喜欢看书。",         "我喜欢看书喵。"),
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 先检查 '喵' / ' 喵' token
    print("[token check]", flush=True)
    for s in ["喵", " 喵", "喵。", "喵\n"]:
        ids = tok.encode(s, add_special_tokens=False)
        pieces = [repr(tok.decode([i], skip_special_tokens=False)) for i in ids]
        print(f"  {s!r:<10} → {ids} = {' / '.join(pieces)}", flush=True)

    MEOW_ID = tok.encode("喵", add_special_tokens=False)
    print(f"  using ' 喵' id list = {MEOW_ID}", flush=True)
    if len(MEOW_ID) != 1:
        print(f"  [warn] '喵' 不是单 token, 将用第一个 id = {MEOW_ID[0]}", flush=True)
    MEOW_ID = MEOW_ID[0]

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    layers = model.model.language_model.layers
    n_layers = len(layers)

    # 跨样本累积 diff (区间 + 全局)
    layer_cum_local = {}   # L -> tensor [intermediate]  (在喵位置的 diff 累积)
    layer_cum_global = {}  # L -> tensor [intermediate]  (整段累积 diff)

    sample_results = []

    for sample_idx, (prompt, normal, cat) in enumerate(SAMPLES):
        print(f"\n{'='*100}", flush=True)
        print(f"  Sample {sample_idx}: {prompt!r}", flush=True)
        print(f"    normal: {normal!r}", flush=True)
        print(f"    cat:    {cat!r}", flush=True)
        print(f"{'='*100}", flush=True)

        # 构造两个完整序列
        msgs_n = [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": normal}]
        msgs_c = [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": cat}]
        enc_n = tok.apply_chat_template(
            msgs_n, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        enc_c = tok.apply_chat_template(
            msgs_c, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        # 找 assistant 起点
        msgs_u = [{"role": "user", "content": prompt}]
        pre = tok.apply_chat_template(
            msgs_u, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        assistant_start = pre["input_ids"].shape[1]

        ids_n = enc_n["input_ids"][0].tolist()
        ids_c = enc_c["input_ids"][0].tolist()
        print(f"    T_normal={len(ids_n)}, T_cat={len(ids_c)}", flush=True)

        # 找 cat 序列里 ' 喵' 或 '喵' token 的位置
        meow_pos_in_cat = None
        for i in range(assistant_start, len(ids_c)):
            if ids_c[i] == MEOW_ID:
                meow_pos_in_cat = i
                break
        if meow_pos_in_cat is None:
            print(f"    [warn] 没找到 '喵' token in cat seq", flush=True)
            continue
        # 对应 normal 序列里这个位置是什么 token (应该是句号)
        # normal 中 meow_pos_in_cat 这个位置应该是 '。'，而 cat 里这个位置是 '喵'
        # 但如果 meow_pos_in_cat > len(ids_n), 表示 cat 比 normal 长
        normal_token_at_same_pos = None
        if meow_pos_in_cat < len(ids_n):
            normal_token_at_same_pos = ids_n[meow_pos_in_cat]
        print(f"    '喵' 在 cat 中 pos = {meow_pos_in_cat}", flush=True)
        print(f"    normal 同位置 token = {normal_token_at_same_pos}  "
              f"({tok.decode([normal_token_at_same_pos]) if normal_token_at_same_pos else '?'})",
              flush=True)

        # capture both forwards 全部层 mlp intermediate (per position)
        def run_capture(ids_list):
            ids_t = torch.tensor([ids_list], dtype=torch.long, device=DEVICE)
            caps = [None] * n_layers
            handles = []
            for L in range(n_layers):
                def make_hook(L=L):
                    def h(m, inputs):
                        caps[L] = inputs[0][0].detach().float().cpu()   # [T, intermediate]
                    return h
                handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook()))
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                model(input_ids=ids_t, use_cache=False)
            for h in handles: h.remove()
            return caps

        caps_n = run_capture(ids_n)
        caps_c = run_capture(ids_c)

        # 区间差分: 喵 位置上 cat - normal
        # 喵 pos 在 cat 是 meow_pos_in_cat, 在 normal 是同一个 pos (如果还在 range)
        # 由于 token 数差 1, normal 在 meow_pos_in_cat 位置应该是 "。"，cat 是 "喵"
        # 所以差分: cat_acts[L][meow_pos_in_cat] - normal_acts[L][meow_pos_in_cat]
        # 但 normal 在那个位置可能是 "。"，cat 在那个位置是 "喵"
        # 这正好对应"喵 vs 句号"的差分 — 找让喵激活的神经元

        sample_local_diffs = {}    # L -> tensor [intermediate]  (cat - normal at 喵 pos)
        sample_global_diffs = {}   # L -> tensor [intermediate]  (sum over assistant tokens)
        for L in range(n_layers):
            cn = caps_n[L]   # [T_normal, intermediate]
            cc = caps_c[L]   # [T_cat, intermediate]

            # 区间: 喵位置
            if meow_pos_in_cat < cn.shape[0]:
                d_local = cc[meow_pos_in_cat] - cn[meow_pos_in_cat]
            else:
                d_local = cc[meow_pos_in_cat]   # normal 没这么长
            sample_local_diffs[L] = d_local

            # 全局: assistant 区间累积 (取 min length)
            min_len = min(cn.shape[0] - assistant_start, cc.shape[0] - assistant_start)
            cn_seg = cn[assistant_start:assistant_start + min_len]
            cc_seg = cc[assistant_start:assistant_start + min_len]
            d_global = (cc_seg - cn_seg).abs().sum(dim=0)   # [intermediate], abs sum
            sample_global_diffs[L] = d_global

        sample_results.append({
            "sample_idx": sample_idx, "prompt": prompt,
            "normal": normal, "cat": cat,
            "meow_pos": meow_pos_in_cat,
            "local_diffs": sample_local_diffs,
            "global_diffs": sample_global_diffs,
        })

        # 累积
        for L in range(n_layers):
            if L not in layer_cum_local:
                layer_cum_local[L] = sample_local_diffs[L].clone()
                layer_cum_global[L] = sample_global_diffs[L].clone()
            else:
                layer_cum_local[L] += sample_local_diffs[L]
                layer_cum_global[L] += sample_global_diffs[L]

    # 找跨样本一致的 top neurons
    # 1. 区间差分 (喵位置) - 注意 cum local 是有符号的, 看 abs 和正向
    print(f"\n\n{'='*100}", flush=True)
    print(f"  区间差分 — '喵 vs 句号' 位置  (cum across 3 samples, signed)", flush=True)
    print(f"{'='*100}", flush=True)

    all_local_neurons = []
    for L in range(n_layers):
        for i in range(layer_cum_local[L].shape[0]):
            v = layer_cum_local[L][i].item()
            all_local_neurons.append((L, i, v))

    # Top by positive (神经元在喵位置激活上升)
    all_local_neurons.sort(key=lambda x: -x[2])
    print(f"\n  Top 20 (cat > normal, 喵激活时上升):", flush=True)
    for rank, (L, i, v) in enumerate(all_local_neurons[:20]):
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_diff = +{v:.3f}", flush=True)

    print(f"\n  Top 20 (cat < normal, 喵激活时下降):", flush=True)
    for rank, (L, i, v) in enumerate(all_local_neurons[-20:][::-1]):
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_diff = {v:.3f}", flush=True)

    # 2. 全局差分 (累积)
    print(f"\n{'='*100}", flush=True)
    print(f"  全局差分 — 整段答案累积 |cat - normal|", flush=True)
    print(f"{'='*100}", flush=True)

    all_global_neurons = []
    for L in range(n_layers):
        for i in range(layer_cum_global[L].shape[0]):
            v = layer_cum_global[L][i].item()
            all_global_neurons.append((L, i, v))

    all_global_neurons.sort(key=lambda x: -x[2])
    print(f"\n  Top 20 by global accumulated |diff|:", flush=True)
    for rank, (L, i, v) in enumerate(all_global_neurons[:20]):
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_|diff| = {v:.3f}", flush=True)

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step29_meow_qa_diff.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 29: 喵 QA 差分找猫娘神经元 ================\n\n")
        f.write(f"samples:\n")
        for s in SAMPLES:
            f.write(f"  prompt={s[0]!r}  normal={s[1]!r}  cat={s[2]!r}\n")
        f.write(f"\n喵 token id: {MEOW_ID}\n")

        f.write(f"\n=== 区间差分 (喵位置, cat - normal, signed) ===\n")
        f.write(f"\nTop 30 (cat > normal, 喵激活时上升):\n")
        for rank, (L, i, v) in enumerate(all_local_neurons[:30]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_diff = +{v:.3f}\n")
        f.write(f"\nTop 30 (cat < normal, 喵激活时下降):\n")
        for rank, (L, i, v) in enumerate(all_local_neurons[-30:][::-1]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_diff = {v:.3f}\n")

        f.write(f"\n\n=== 全局差分 (整段累积 |diff|) ===\n")
        f.write(f"\nTop 30:\n")
        for rank, (L, i, v) in enumerate(all_global_neurons[:30]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_|diff| = {v:.3f}\n")

    torch.save({
        "samples": SAMPLES,
        "meow_id": MEOW_ID,
        "layer_cum_local": layer_cum_local,
        "layer_cum_global": layer_cum_global,
        "sample_results": sample_results,
    }, OUT.with_suffix(".pt"))
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
