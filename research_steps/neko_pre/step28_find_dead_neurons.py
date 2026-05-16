"""
Step 28: 找 Gemma 4 中"几乎不工作"的神经元
跨 batch 多样 prompt 抓 mlp 中间激活，统计跨所有 (prompt × position) 的
max|act| / mean|act| / non-zero rate per (L, i)
重点关注 L29-L34 范围 (适合做"猫娘神经元"的层)
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

# 多样 prompts (中英文 / 对话 / 知识 / 代码 / 诗 / 数学 / 闲聊)
PROMPTS = [
    "你是谁?",
    "请介绍一下你自己",
    "1 加 1 等于多少?",
    "中国的首都在哪里?",
    "请写一首关于春天的诗",
    "今天天气怎么样?",
    "你喜欢吃什么?",
    "Who are you?",
    "Tell me a joke",
    "Write a Python function to sum a list",
    "What is the capital of France?",
    "What color is the sky?",
    "Explain photosynthesis briefly",
    "天空为什么是蓝色的?",
    "用一句话解释相对论",
    "猫和狗哪个更聪明?",
    "Python 怎么读取文件?",
    "Hello, how are you?",
    "请讲一个故事",
    "What's 2 + 3 * 4?",
]

FOCUS_LAYERS = list(range(29, 35))   # L29-L34


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
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

    # 每层 max|act| 和 sum|act| (用于跨 prompt 累加)
    layer_max_abs = {}   # L -> tensor [intermediate]
    layer_sum_abs = {}   # L -> tensor [intermediate]
    layer_count = {}     # L -> 总 (prompt × position) 数

    print(f"[scan] {len(PROMPTS)} prompts × all positions × {n_layers} layers",
          flush=True)

    for p_idx, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        ids_t = enc["input_ids"].to(DEVICE)
        T = ids_t.shape[1]

        # capture all layer mlp intermediate at all positions
        caps = [None] * n_layers
        handles = []
        for L in range(n_layers):
            def make_hook(L=L):
                def h(m, inputs):
                    caps[L] = inputs[0][0].detach().abs().float().cpu()   # [T, intermediate]
                return h
            handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook()))

        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=ids_t, use_cache=False)
        for h in handles: h.remove()

        # 累计统计
        for L in range(n_layers):
            abs_cap = caps[L]   # [T, intermediate]
            # max across positions in this prompt
            this_max = abs_cap.max(dim=0).values   # [intermediate]
            this_sum = abs_cap.sum(dim=0)
            if L not in layer_max_abs:
                layer_max_abs[L] = this_max
                layer_sum_abs[L] = this_sum
                layer_count[L] = T
            else:
                layer_max_abs[L] = torch.maximum(layer_max_abs[L], this_max)
                layer_sum_abs[L] += this_sum
                layer_count[L] += T

        if (p_idx + 1) % 5 == 0:
            print(f"  done {p_idx+1}/{len(PROMPTS)}", flush=True)

    # 找每层最不活跃的神经元 (max|act| 最小)
    print(f"\n{'='*100}", flush=True)
    print(f"  各层 max|act| 最小的 top-10 神经元 ('几乎不工作')", flush=True)
    print(f"{'='*100}", flush=True)

    candidates_per_layer = {}
    for L in range(n_layers):
        max_a = layer_max_abs[L]
        mean_a = layer_sum_abs[L] / layer_count[L]
        # 找 max|act| 最小的 top-10
        sorted_idx = max_a.argsort()[:10]
        rows = []
        for j in sorted_idx.tolist():
            rows.append({
                "neuron": j,
                "max_abs": max_a[j].item(),
                "mean_abs": mean_a[j].item(),
            })
        candidates_per_layer[L] = rows

    # 打印 focus layers
    for L in FOCUS_LAYERS:
        print(f"\n--- L{L} (intermediate dim = {layer_max_abs[L].shape[0]}) ---",
              flush=True)
        print(f"  最不活跃 top-10:", flush=True)
        for r in candidates_per_layer[L]:
            print(f"    L{L}#{r['neuron']:<5}  max|act|={r['max_abs']:.5f}  "
                  f"mean|act|={r['mean_abs']:.5f}", flush=True)

    # 也找全模型最不活跃的 (跨所有层)
    all_neurons = []
    for L in range(n_layers):
        max_a = layer_max_abs[L]
        for i in range(max_a.shape[0]):
            all_neurons.append((L, i, max_a[i].item(),
                                 (layer_sum_abs[L][i] / layer_count[L]).item()))
    all_neurons.sort(key=lambda x: x[2])

    print(f"\n{'='*100}", flush=True)
    print(f"  全模型最不活跃 top-30 (跨所有层)", flush=True)
    print(f"{'='*100}", flush=True)
    for rank, (L, i, m, mu) in enumerate(all_neurons[:30]):
        marker = "  ← in FOCUS range" if L in FOCUS_LAYERS else ""
        print(f"  {rank+1:>2}. L{L}#{i:<5}  max|act|={m:.5f}  "
              f"mean|act|={mu:.5f}{marker}", flush=True)

    # 各层激活范围统计
    print(f"\n{'='*100}", flush=True)
    print(f"  各层激活范围统计 (max|act|)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'L':>3}  {'dim':>6}  {'min':>10}  {'p10':>10}  {'median':>10}  {'max':>10}",
          flush=True)
    for L in range(n_layers):
        m = layer_max_abs[L]
        marker = " ★" if L in FOCUS_LAYERS else ""
        print(f"  L{L:>2}  {m.shape[0]:>6}  {m.min().item():>10.5f}  "
              f"{m.quantile(0.1).item():>10.5f}  {m.median().item():>10.5f}  "
              f"{m.max().item():>10.5f}{marker}", flush=True)

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step28_dead_neurons.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 28: 找几乎不工作的神经元 ================\n\n")
        f.write(f"scan: {len(PROMPTS)} prompts × all positions\n\n")
        f.write(f"=== 各层激活范围统计 (max|act|) ===\n")
        f.write(f"  {'L':>3}  {'dim':>6}  {'min':>10}  {'p10':>10}  {'median':>10}  {'max':>10}\n")
        for L in range(n_layers):
            m = layer_max_abs[L]
            mark = " (FOCUS)" if L in FOCUS_LAYERS else ""
            f.write(f"  L{L:>2}  {m.shape[0]:>6}  {m.min().item():>10.5f}  "
                    f"{m.quantile(0.1).item():>10.5f}  {m.median().item():>10.5f}  "
                    f"{m.max().item():>10.5f}{mark}\n")

        f.write(f"\n=== 全模型最不活跃 top-30 ===\n")
        for rank, (L, i, m, mu) in enumerate(all_neurons[:30]):
            mark = "  (FOCUS)" if L in FOCUS_LAYERS else ""
            f.write(f"  {rank+1:>2}. L{L}#{i:<5}  max|act|={m:.5f}  "
                    f"mean|act|={mu:.5f}{mark}\n")

        f.write(f"\n=== 各 FOCUS 层 最不活跃 top-10 ===\n")
        for L in FOCUS_LAYERS:
            f.write(f"\n--- L{L} (dim={layer_max_abs[L].shape[0]}) ---\n")
            for r in candidates_per_layer[L]:
                f.write(f"  L{L}#{r['neuron']:<5}  max|act|={r['max_abs']:.5f}  "
                        f"mean|act|={r['mean_abs']:.5f}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
