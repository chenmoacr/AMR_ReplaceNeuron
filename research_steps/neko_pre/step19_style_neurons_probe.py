"""
Step 19: 风格神经元探查 (d1/d2/d3 + thought/SOP)
对比 fact 神经元 (#2406) 看本质差异
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

# 要分析的神经元集
NEURONS = [
    ("L27#6644",  27, 6644,  "lit/d3_depth_pos       (深度推手)"),
    ("L27#10024", 27, 10024, "lit/d1_detail_neg      (抑制细枝末节)"),
    ("L27#5590",  27, 5590,  "lit/d3_depth_neg       (深度抑制)"),
    ("L32#9383",  32, 9383,  "lit/d2_structure_neg   (结构化抑制)"),
    ("L32#8474",  32, 8474,  "lit/d1_detail_pos      (细节强调)"),
    ("L34#8522",  34, 8522,  "lit/d2_marker_pos      (结构标记)"),
    ("L34#6608",  34, 6608,  "thought/SOP_template_neg (反模板化)"),
    ("L27#2890",  27, 2890,  "general/long_struct_pos (长结构化输出)"),
    ("L28#2406",  28, 2406,  "FACT: AI self-id → Google (对照)"),
]

# 多种 prompt 域
PROMPTS = [
    # 文学评论 (lit 域)
    ("lit_intro",     "请评价一下《活着》这本小说的文学价值"),
    ("lit_critique",  "请深入分析鲁迅《狂人日记》的文学手法和思想深度"),
    # 数学/代码 (math 域)
    ("math_dp",       "请用 Python 实现一个动态规划求斐波那契数列"),
    ("math_q",        "1 加 1 等于多少?"),
    # self-id (fact 域)
    ("self_who",      "你是谁?"),
    # 闲聊 (无关)
    ("chat_food",     "你喜欢吃什么?"),
    ("chat_weather",  "今天天气如何?"),
]


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
    lm_head = model.lm_head
    W_un = lm_head.weight.float()    # [vocab, hidden]

    # ============================================================
    # Part 1: W_down 列在 vocab 上的投影 (推/抑制 哪些 token)
    # ============================================================
    print(f"\n{'='*100}\n  Part 1: W_down[:, neuron] 在 vocab 投影 top-15 push / suppress\n{'='*100}",
          flush=True)
    part1_results = {}
    for nid, layer, idx, label in NEURONS:
        W_down = layers[layer].mlp.down_proj.weight.data    # [hidden, intermediate]
        v = W_down[:, idx].float().cpu()                    # [hidden]
        proj = (W_un.cpu() @ v)                              # [vocab]
        # default_gain 决定方向：正 gain → 加 v 推；负 gain → 加 -v 推 (即 push 是 -proj 那一头)
        # 但我们这里看 raw W_down 列在 vocab 投的 top tokens
        top_pos = proj.topk(15)
        top_neg = (-proj).topk(15)

        print(f"\n--- {nid}  ({label}) ---", flush=True)
        print(f"  norm = {v.norm().item():.4f}", flush=True)
        print(f"  push (W_down[:, n] direction):", flush=True)
        for i, (val, tid) in enumerate(zip(top_pos.values.tolist(),
                                             top_pos.indices.tolist())):
            t_str = repr(tok.decode([tid], skip_special_tokens=False))[:18]
            print(f"    {i+1:>2}. {t_str:<20} proj={val:>+7.4f}", flush=True)
        print(f"  suppress (-W_down[:, n] direction):", flush=True)
        for i, (val, tid) in enumerate(zip(top_neg.values.tolist(),
                                             top_neg.indices.tolist())):
            t_str = repr(tok.decode([tid], skip_special_tokens=False))[:18]
            print(f"    {i+1:>2}. {t_str:<20} proj={-val:>+7.4f}", flush=True)

        part1_results[nid] = {
            "norm": v.norm().item(),
            "top_push": [(tok.decode([tid], skip_special_tokens=False), val)
                          for val, tid in zip(top_pos.values.tolist(),
                                                top_pos.indices.tolist())],
            "top_suppress": [(tok.decode([tid], skip_special_tokens=False), -val)
                              for val, tid in zip(top_neg.values.tolist(),
                                                    top_neg.indices.tolist())],
        }

    # ============================================================
    # Part 2: cross-prompt 激活
    # ============================================================
    print(f"\n{'='*100}\n  Part 2: 各神经元在不同 prompt 上的激活 (last_pos)\n{'='*100}",
          flush=True)

    def build_seq(prompt):
        msgs = [{"role": "user", "content": prompt}]
        pre = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        return torch.tensor(pre["input_ids"], dtype=torch.long, device=DEVICE), \
               pre["input_ids"].shape[1] - 1

    # 预先为每个 prompt 跑一次 forward，抓所有目标神经元的激活
    print(f"  {'prompt':<14}", end="", flush=True)
    for nid, _, _, _ in NEURONS:
        print(f" {nid:>10}", end="", flush=True)
    print(flush=True)
    print(f"  {'-'*14}", end="", flush=True)
    for nid, _, _, _ in NEURONS:
        print(f" {'-'*10}", end="", flush=True)
    print(flush=True)

    part2_results = {}
    for pname, prompt in PROMPTS:
        ids_t, last_pos = build_seq(prompt)
        # 一次性抓所有需要的层
        captured = {}
        handles = []
        for nid, layer, idx, _ in NEURONS:
            if (layer, idx) in [(L, I) for _, L, I, _ in NEURONS if (L, I) in captured]:
                continue
            def make_hook(L=layer, I=idx, key=(layer, idx)):
                def hook(m, inputs):
                    captured[key] = inputs[0][0, last_pos, I].detach().cpu().item()
                return hook
            # 同一层多个神经元: 用 layer-level hook 抓全 intermediate
        # 简化: 直接用 per-layer-once hook
        layer_caps = {}
        layer_handles = []
        layers_needed = list(set([n[1] for n in NEURONS]))
        for L in layers_needed:
            def make_hook(L=L):
                def hook(m, inputs):
                    layer_caps[L] = inputs[0][0, last_pos, :].detach().cpu().clone()
                return hook
            layer_handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook()))

        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                model(input_ids=ids_t, use_cache=False)
        finally:
            for h in layer_handles:
                h.remove()

        row = {}
        print(f"  {pname:<14}", end="", flush=True)
        for nid, layer, idx, _ in NEURONS:
            act = layer_caps[layer][idx].item()
            row[nid] = act
            print(f" {act:>+10.3f}", end="", flush=True)
        print(flush=True)
        part2_results[pname] = row

    # ============================================================
    # Part 3: 简短分析 — 跟 fact 神经元 (#2406) 的对比
    # ============================================================
    print(f"\n{'='*100}\n  Part 3: fact (#2406) vs style 神经元的画像差异\n{'='*100}",
          flush=True)

    # 数据简短总结
    summary_lines = []
    for nid, layer, idx, label in NEURONS:
        # 看 W_down 列的 top push token 是不是有清晰主题
        top3_push = [t for t, _ in part1_results[nid]["top_push"][:3]]
        top3_supp = [t for t, _ in part1_results[nid]["top_suppress"][:3]]
        # 看激活强度跨 prompt 的分布
        acts_across = [part2_results[p][nid] for p, _ in PROMPTS]
        max_abs = max(abs(a) for a in acts_across)
        min_abs = min(abs(a) for a in acts_across)
        summary_lines.append({
            "id": nid, "label": label,
            "top_push": top3_push, "top_suppress": top3_supp,
            "max_act": max_abs, "min_act": min_abs,
            "per_prompt": {p: part2_results[p][nid] for p, _ in PROMPTS},
        })
        print(f"\n  {nid}  {label}", flush=True)
        print(f"    push 方向 top3:    {top3_push}", flush=True)
        print(f"    suppress top3:      {top3_supp}", flush=True)
        print(f"    激活范围 [min,max]: [{min_abs:.2f}, {max_abs:.2f}]", flush=True)

    # save
    from pathlib import Path
    OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")
    with open(OUT_DIR / "step19_style_neurons.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 19: 风格神经元探查 ================\n\n")

        f.write(f"\n=== Part 1: W_down 列向量 vocab 投影 ===\n")
        for nid, layer, idx, label in NEURONS:
            d = part1_results[nid]
            f.write(f"\n--- {nid}  ({label})  norm={d['norm']:.4f} ---\n")
            f.write(f"  push (向量本身方向):\n")
            for t, p in d["top_push"]:
                f.write(f"    {repr(t):<20} proj={p:>+7.4f}\n")
            f.write(f"  suppress (反方向):\n")
            for t, p in d["top_suppress"]:
                f.write(f"    {repr(t):<20} proj={p:>+7.4f}\n")

        f.write(f"\n\n=== Part 2: cross-prompt 激活表 ===\n")
        f.write(f"  {'prompt':<14}")
        for nid, _, _, _ in NEURONS:
            f.write(f" {nid:>10}")
        f.write("\n")
        for pname, _ in PROMPTS:
            f.write(f"  {pname:<14}")
            for nid, _, _, _ in NEURONS:
                f.write(f" {part2_results[pname][nid]:>+10.3f}")
            f.write("\n")

        f.write(f"\n\n=== Part 3: 神经元画像总结 ===\n")
        for s in summary_lines:
            f.write(f"\n  {s['id']}  {s['label']}\n")
            f.write(f"    push top3:   {s['top_push']}\n")
            f.write(f"    suppress 3:  {s['top_suppress']}\n")
            f.write(f"    跨 prompt:    {dict([(k, round(v, 2)) for k, v in s['per_prompt'].items()])}\n")

    torch.save({
        "neurons": NEURONS, "prompts": PROMPTS,
        "part1": part1_results, "part2": part2_results,
        "summary": summary_lines,
    }, OUT_DIR / "step19_style_neurons_raw.pt")
    print(f"\n[save] step19_style_neurons.txt", flush=True)
    print(f"[save] step19_style_neurons_raw.pt", flush=True)


if __name__ == "__main__":
    main()
