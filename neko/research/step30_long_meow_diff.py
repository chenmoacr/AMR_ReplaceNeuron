"""
Step 30: 长答案 + 多喵 QA 差分

Step 1: 多 prompt 跑 baseline gen (greedy, max_new=2048)
Step 2: 自动在每个句号前插入 '喵' 生成 catmeowline
Step 3: 存 JSON (input/baseline/catmeowline)
Step 4: 跑两次 teacher-force 抓 mlp intermediate
Step 5: 在每个喵位置做差分, 跨 prompt 累积找猫娘神经元
"""
import os, sys, json, re
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
MAX_NEW = 2048

PROMPTS = [
    "洛杉矶常年天气怎么样?",
    "当你不开心的时候你会怎么做?",
    "Google 是个怎样的公司?",
    "猕猴桃是什么?",
]


def get_eos_ids(tok):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    return list({i for i in eos if i is not None})


def insert_meow(text):
    """在每个句号 '。' 或 '.' 之前插入 '喵'"""
    # 用 regex 处理多种句末标点
    # 只替换"中文句号"和"英文句号", 避免影响数字小数点
    result = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '。':
            result.append('喵')
            result.append(c)
        elif c == '.':
            # 避免误伤数字小数点 (如 3.14)
            # 简单规则: 前一个字符不是数字, 后一个字符不是数字
            prev_ch = text[i-1] if i > 0 else ' '
            next_ch = text[i+1] if i + 1 < len(text) else ' '
            if (not prev_ch.isdigit()) and (not next_ch.isdigit()):
                result.append('喵')
                result.append(c)
            else:
                result.append(c)
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)

    MEOW_IDS = tok.encode("喵", add_special_tokens=False)
    if len(MEOW_IDS) != 1:
        print(f"[err] '喵' 不是单 token: {MEOW_IDS}", flush=True)
        return
    MEOW_ID = MEOW_IDS[0]
    print(f"[token] '喵' id = {MEOW_ID}", flush=True)

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

    # ============================================================
    # Step 1+2+3: gen baseline → 生成 catmeowline → 存 JSON
    # ============================================================
    print(f"\n{'='*100}\n  Step 1-3: 跑 baseline gen + 构造 catmeowline + 存 JSON\n{'='*100}",
          flush=True)

    samples = []
    for idx, prompt in enumerate(PROMPTS):
        print(f"\n[gen] {idx}: {prompt!r}", flush=True)
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = out[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        baseline = tok.decode(gen_ids, skip_special_tokens=True)
        catmeowline = insert_meow(baseline)

        # 统计喵数量
        n_meow = catmeowline.count("喵")
        print(f"  baseline len: {len(baseline)} chars, {len(gen_ids)} tokens",
              flush=True)
        print(f"  n_meow inserted: {n_meow}", flush=True)
        print(f"  baseline (head): {baseline[:200]}", flush=True)
        print(f"  catmeow  (head): {catmeowline[:200]}", flush=True)

        samples.append({
            "idx": idx,
            "input": prompt,
            "baseline": baseline,
            "catmeowline": catmeowline,
            "n_meow": n_meow,
        })

    # 存 JSON
    json_path = Path("J:/amr/amr_wtf/identity_swap/step30_samples.json")
    json_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n[save json] {json_path}", flush=True)

    # ============================================================
    # Step 4+5: QA 差分
    # ============================================================
    print(f"\n{'='*100}\n  Step 4-5: QA 差分 — 抓 mlp intermediate + 喵位置累积\n{'='*100}",
          flush=True)

    # 累积统计
    layer_cum_local = {}      # L -> tensor [intermediate] (cat-base @ 喵位置 累积)
    layer_count_meows = 0      # 累积喵数 (用于平均)

    all_top_neurons_per_sample = []

    for s in samples:
        prompt = s["input"]
        baseline_text = s["baseline"]
        catmeow_text = s["catmeowline"]

        # 构造 sequences
        msgs_b = [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": baseline_text}]
        msgs_c = [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": catmeow_text}]
        enc_b = tok.apply_chat_template(
            msgs_b, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        enc_c = tok.apply_chat_template(
            msgs_c, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        ids_b = enc_b["input_ids"][0].tolist()
        ids_c = enc_c["input_ids"][0].tolist()
        T_b = len(ids_b)
        T_c = len(ids_c)
        print(f"\n[sample {s['idx']}] {prompt!r}  T_b={T_b}, T_c={T_c}", flush=True)

        # 找 cat 中所有 '喵' 位置
        meow_positions = [i for i, t in enumerate(ids_c) if t == MEOW_ID]
        print(f"  meow positions in cat: {meow_positions[:20]}... 共 {len(meow_positions)} 个",
              flush=True)
        if not meow_positions:
            print(f"  [warn] no meow tokens, skip", flush=True)
            continue

        # forward + capture
        def run_capture(ids_list, T):
            ids_t = torch.tensor([ids_list], dtype=torch.long, device=DEVICE)
            caps = [None] * n_layers
            handles = []
            for L in range(n_layers):
                def make_hook(L=L):
                    def h(m, inputs):
                        caps[L] = inputs[0][0].detach().float().cpu()  # [T, intermediate]
                    return h
                handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook()))
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                model(input_ids=ids_t, use_cache=False)
            for h in handles: h.remove()
            return caps

        print(f"  forward baseline (T={T_b}) ...", flush=True)
        caps_b = run_capture(ids_b, T_b)
        print(f"  forward catmeow (T={T_c}) ...", flush=True)
        caps_c = run_capture(ids_c, T_c)

        # 在每个 cat 中的喵位置, baseline 在同一 pos (如果还在 range) 是什么 token?
        # 这里关键: baseline 和 cat 在前面 K 位置都是相同 (chat header + prompt + answer 同 prefix)
        # 但 cat 中插入了喵, 所以从第一个喵开始, cat 的 token 序列就跟 baseline 偏移
        # 即 cat[meow_pos[0]] = 喵, baseline[meow_pos[0]] = 句号 (因为他们对齐到同一段 hidden 处理深度)
        # 我们想找的是 "在喵位置, cat 的激活 - baseline 在同 token 位置的激活"
        #
        # 严格地, 应该用每个喵 m_k 对应的 baseline 中"原本是句号"的位置
        # cat 中第 k 个喵后, cat 和 baseline 的 token 位置偏移 k (cat 多 k 个)
        # 所以 baseline 中对应"那个句号"的位置 = m_k - k
        # 严格说在 cat 中 meow_positions[k] (第 k 个喵) 对应 baseline 中的 (meow_positions[k] - k)

        sample_local_diff = {L: torch.zeros(caps_b[L].shape[1]) for L in range(n_layers)}
        valid_meows = 0
        for k, m_pos_c in enumerate(meow_positions):
            m_pos_b = m_pos_c - k   # baseline 中"原本是句号"的位置
            if m_pos_b < 0 or m_pos_b >= T_b: continue
            for L in range(n_layers):
                d = caps_c[L][m_pos_c] - caps_b[L][m_pos_b]
                sample_local_diff[L] += d
            valid_meows += 1
        print(f"  valid meow pairs: {valid_meows}", flush=True)

        # 累积
        for L in range(n_layers):
            if L in layer_cum_local:
                layer_cum_local[L] += sample_local_diff[L]
            else:
                layer_cum_local[L] = sample_local_diff[L].clone()
        layer_count_meows += valid_meows

        # 顺便记录单样本的 top
        sample_top = []
        for L in range(n_layers):
            for i in range(sample_local_diff[L].shape[0]):
                sample_top.append((L, i, sample_local_diff[L][i].item()))
        sample_top.sort(key=lambda x: -abs(x[2]))
        all_top_neurons_per_sample.append(sample_top[:30])

        # 释放 capture
        del caps_b, caps_c
        torch.cuda.empty_cache()

    # ============================================================
    # 跨样本 top 神经元
    # ============================================================
    print(f"\n\n{'='*100}", flush=True)
    print(f"  跨 {len(samples)} 个 prompt × {layer_count_meows} 喵位置, 累积差分", flush=True)
    print(f"{'='*100}", flush=True)

    all_neurons = []
    for L in range(n_layers):
        for i in range(layer_cum_local[L].shape[0]):
            all_neurons.append((L, i, layer_cum_local[L][i].item()))

    # 上升 (cat > base, 喵激活时上升)
    all_neurons.sort(key=lambda x: -x[2])
    print(f"\n  Top 30 (喵位置激活上升):", flush=True)
    for rank, (L, i, v) in enumerate(all_neurons[:30]):
        mark = " ★ mid-late" if L >= 25 else ""
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_diff = +{v:.3f}{mark}", flush=True)

    # 下降
    print(f"\n  Top 30 (喵位置激活下降):", flush=True)
    for rank, (L, i, v) in enumerate(all_neurons[-30:][::-1]):
        mark = " ★ mid-late" if L >= 25 else ""
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_diff = {v:.3f}{mark}", flush=True)

    # 仅 mid-late (L25+)
    mid_late = [n for n in all_neurons if n[0] >= 25]
    print(f"\n  Mid-late (L25+) Top 20 上升:", flush=True)
    for rank, (L, i, v) in enumerate(mid_late[:20]):
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_diff = +{v:.3f}", flush=True)
    print(f"\n  Mid-late (L25+) Top 20 下降:", flush=True)
    for rank, (L, i, v) in enumerate(mid_late[-20:][::-1]):
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_diff = {v:.3f}", flush=True)

    # save
    OUT = Path("J:/amr/amr_wtf/identity_swap/step30_meow_diff.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 30: 长答案 多喵 QA 差分 ================\n\n")
        f.write(f"samples: {len(samples)}, total meows: {layer_count_meows}\n\n")
        for s in samples:
            f.write(f"\n[{s['idx']}] {s['input']!r}\n")
            f.write(f"  baseline ({len(s['baseline'])}c): {s['baseline'][:300]}...\n")
            f.write(f"  catmeow:  {s['catmeowline'][:300]}...\n")

        f.write(f"\n\n=== 全部 35 层 — 累积差分 top 30 (上升) ===\n")
        for rank, (L, i, v) in enumerate(all_neurons[:30]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_diff = +{v:.3f}\n")
        f.write(f"\n=== 全部 35 层 — 累积差分 top 30 (下降) ===\n")
        for rank, (L, i, v) in enumerate(all_neurons[-30:][::-1]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_diff = {v:.3f}\n")

        f.write(f"\n\n=== Mid-late (L25+) Top 30 (上升) ===\n")
        for rank, (L, i, v) in enumerate(mid_late[:30]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_diff = +{v:.3f}\n")
        f.write(f"\n=== Mid-late (L25+) Top 30 (下降) ===\n")
        for rank, (L, i, v) in enumerate(mid_late[-30:][::-1]):
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  cum_diff = {v:.3f}\n")
    torch.save({
        "samples": samples,
        "layer_cum_local": layer_cum_local,
        "layer_count_meows": layer_count_meows,
        "top_neurons": all_neurons[:200],
    }, OUT.with_suffix(".pt"))
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
