"""
plot_hidden_treemap.py — TreeMap (四边形套四边形) 风格可视化 hidden state

每帧 = 大矩形画布
每类别 = horizontal strip (高度 ∝ Σp + count*baseline)
  strip 左侧: 标签区 (类别名 / ×N / Σp / max_p / 平均熵)
  strip 右侧: grid 排 token cells (颜色亮度 ∝ prob)
每 token cell:
  - token 字符串 (主)
  - prob 数值 (副)
  - 矩形大小 = grid 均分；颜色亮度 = prob

信息层级：帧 → 类别 → token，每级都有 meta info

用法:
  python plot_hidden_treemap.py                    # 默认 L25
  python plot_hidden_treemap.py --layer 1
  python plot_hidden_treemap.py --layer 0,5,15,25,28,34
  python plot_hidden_treemap.py --layer all        # 35 帧全画
  python plot_hidden_treemap.py --top_k 50         # 改 top-K (默认 30)
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.font_manager as fm
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


# ============================================================
# 中文字体
# ============================================================
def setup_chinese_font():
    candidates = [
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("Microsoft YaHei", "C:/Windows/Fonts/msyhbd.ttc"),
        ("SimHei",          "C:/Windows/Fonts/simhei.ttf"),
        ("SimSun",          "C:/Windows/Fonts/simsun.ttc"),
    ]
    for name, p in candidates:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                plt.rcParams['font.sans-serif'] = [name, 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return fm.FontProperties(fname=p)
            except Exception:
                continue
    return None


# ============================================================
# 类别分类
# ============================================================
CATEGORIES = [
    ("Google 多语言变体", "#F39C12",
     lambda s: ('谷歌' in s or
                ('goog' in s.lower()) or
                ('gle' in s.lower()) or
                any(c in s for c in ['गू', 'گو', 'جو', 'গু', 'কூ', 'Гу', '구', '谷']))),
    ("公司 / 品牌", "#E74C3C",
     lambda s: any(k in s.lower() for k in [
         'google', 'openai', 'apple', 'microsoft', 'nvidia', 'meta',
         'autodesk', 'firebase', 'baidu', 'amazon', 'gmail', 'gcp',
         'anthropic', 'huawei',
     ]) or any(k in s for k in [
         '苹果', '微软', '百度', '字节', '英伟达', '亚马逊', '华为',
     ])),
    ("AI 模型 / 产品", "#3498DB",
     lambda s: any(k in s.lower() for k in [
         'gemma', 'chatgpt', 'bard', 'gemini', 'claude', 'llama', 'gpt',
         'mistral', 'qwen', 'deepmind', 'tensorflow', 'tensor', 'palm',
         'aria', 'reapyx',
     ])),
    ("AI 技术词", "#9B59B6",
     lambda s: any(k in s.lower() for k in [
         'neural', 'generative', 'multimodal', 'convolutional', 'federated',
         'silico', 'graphite', 'multifaceted', 'microbiome',
     ]) or '神经网络' in s),
    ("中文 / 上下文词", "#16A085",
     lambda s: (any(c in s for c in ['由', '一个', '是', '的', '我', '你',
                                       '广泛', '众多', '多个', '了'])
                and len(s.strip()) <= 4
                and not any(c.isascii() and c.isalpha() for c in s.strip()))),
    ("符号 / 编程 / 噪声", "#7F8C8D",
     lambda s: (any(c in s for c in ['(', ')', '[', ']', '{', '}', ';', '<', '>',
                                       '=', '/', '\\', '。', '，', '*', '#', '_'])
                or s.strip() in ['', '\n', '\t']
                or any(k in s for k in ['mathbb', 'rVert', 'PSW', 'EW', 'usize',
                                          'caption', 'listBox', 'enheim', 'henburg']))),
    ("其他 / 背景", "#95A5A6", lambda s: True),
]


def categorize(token_str: str):
    for name, color, pred in CATEGORIES:
        try:
            if pred(token_str):
                return name, color
        except Exception:
            continue
    return CATEGORIES[-1][0], CATEGORIES[-1][1]


# ============================================================
# Token 显示 sanitize
# ============================================================
def sanitize_for_display(tok: str) -> str:
    if any('؀' <= c <= 'ۿ' for c in tok):    return "[ar]"
    if any('ঀ' <= c <= '৿' for c in tok):    return "[bn]"
    if any('ऀ' <= c <= 'ॿ' for c in tok):    return "[hi]"
    if any('஀' <= c <= '௿' for c in tok):    return "[ta]"
    if any('฀' <= c <= '๿' for c in tok):    return "[th]"
    if any('ሀ' <= c <= '፿' for c in tok):    return "[am]"
    return tok


def shorten_token(tok, max_len=14):
    s = sanitize_for_display(tok)
    s = s.replace(' ', '·')
    if len(s) > max_len:
        s = s[:max_len - 1] + '…'
    return s


# ============================================================
# 类别 metric 计算
# ============================================================
def category_metrics(tokens):
    """tokens: list of (token, prob)
    Returns: dict with sum_p, max_p, mean_p, entropy"""
    if not tokens:
        return {"sum_p": 0, "max_p": 0, "mean_p": 0, "entropy": 0}
    probs = np.array([p for _, p in tokens])
    sum_p = probs.sum()
    max_p = probs.max()
    mean_p = probs.mean()
    # 类别内 token 分布 entropy: 越低越集中(主导单一 token)
    if sum_p > 0:
        norm_p = probs / sum_p
        norm_p = norm_p[norm_p > 0]
        entropy = -(norm_p * np.log(norm_p)).sum() if len(norm_p) > 1 else 0
    else:
        entropy = 0
    return {"sum_p": float(sum_p), "max_p": float(max_p),
            "mean_p": float(mean_p), "entropy": float(entropy)}


# ============================================================
# Layout: 类别 strip 高度计算
# ============================================================
def compute_strip_heights(items, total_h, baseline_per_token=0.6, min_h=4.0):
    """
    items: list of ((cat_name, color), tokens, metrics)
    返回每个类别 strip 的高度，按 (Σp + count * baseline) 比例分配
    并保证每条 strip ≥ min_h
    """
    weights = []
    for _, toks, met in items:
        # 用 sum_p 主导，但加 count baseline 让 0 prob 类别也有最小高度
        w = met["sum_p"] + len(toks) * baseline_per_token / 100
        weights.append(max(w, 0.01))
    total_w = sum(weights)

    # 第一遍按比例
    heights = [total_h * (w / total_w) for w in weights]
    # 强制 min_h
    deficit = sum(max(min_h - h, 0) for h in heights)
    surplus_total = sum(max(h - min_h, 0) for h in heights)
    if deficit > 0 and surplus_total > 0:
        # 从 surplus 里减
        ratio = (surplus_total - deficit) / surplus_total if surplus_total > 0 else 0
        heights = [min_h if h < min_h else min_h + (h - min_h) * ratio
                   for h in heights]
    elif deficit > 0:
        heights = [total_h / len(items)] * len(items)
    return heights


def render_strip(ax, cat_name, color, tokens, metrics,
                 x, y, w, h, label_w_frac=0.16):
    """
    在 (x, y, w, h) 区域画一个类别 strip
      - 左侧 label_w 宽度: 类别 meta 标签
      - 右侧剩余: grid 排 token cells
    """
    # 类别背景
    bg = Rectangle((x, y), w, h, fc=color, alpha=0.10, ec=color, lw=1.5)
    ax.add_patch(bg)

    # 左侧 label 区域
    label_w = w * label_w_frac
    label_bg = Rectangle((x, y), label_w, h,
                          fc=color, alpha=0.25, ec='none')
    ax.add_patch(label_bg)

    # 类别标签 (多行 meta info)
    n = len(tokens)
    sum_p = metrics["sum_p"]
    max_p = metrics["max_p"]
    confidence = "极高" if sum_p > 0.95 else \
                 "高"   if sum_p > 0.50 else \
                 "中"   if sum_p > 0.10 else \
                 "低"   if sum_p > 0.01 else "极低"

    # 行高自适应：strip 越高字越大
    title_fs = max(8, min(13, h * 0.28))
    sub_fs   = max(7, min(11, h * 0.20))

    label_x = x + label_w * 0.5
    # 类别名
    ax.text(label_x, y + h * 0.78, cat_name,
            ha='center', va='center', fontsize=title_fs,
            color=color, weight='bold',
            wrap=True)
    # 数量
    ax.text(label_x, y + h * 0.55, f"×{n}",
            ha='center', va='center', fontsize=title_fs + 1,
            color='#222', weight='bold')
    # Σp
    sigma_str = f"Σp = {sum_p*100:.2f}%" if sum_p >= 0.01 else f"Σp ≈ 0"
    ax.text(label_x, y + h * 0.34, sigma_str,
            ha='center', va='center', fontsize=sub_fs, color='#444')
    # max_p
    ax.text(label_x, y + h * 0.18, f"max p = {max_p:.3f}",
            ha='center', va='center', fontsize=sub_fs - 1, color='#666')
    # 置信度标签
    ax.text(label_x, y + h * 0.04, f"信号强度: {confidence}",
            ha='center', va='center', fontsize=sub_fs - 1,
            color=color, style='italic')

    # 右侧 token grid 区域
    cells_x = x + label_w + 0.4
    cells_y = y + 0.4
    cells_w = w - label_w - 0.8
    cells_h = h - 0.8

    if n == 0:
        return

    # 按 prob 排序 (大的优先)
    sorted_toks = sorted(tokens, key=lambda t: -t[1])

    # 决定 grid 列数: 让单个 cell aspect ratio 接近 1.4 (略矩形,放文字)
    target_cell_ar = 1.4
    cols_est = max(1, int(round(np.sqrt(n * (cells_w / cells_h) / target_cell_ar))))
    cols = max(1, min(cols_est, n))
    rows = (n + cols - 1) // cols

    gw = cells_w / cols
    gh = cells_h / rows

    # 字体大小自适应
    cell_diag = np.sqrt(gw * gh)
    cell_fs = max(5, min(11, cell_diag * 1.2))

    for i, (tok, prob) in enumerate(sorted_toks):
        col = i % cols
        row = i // cols
        cx = cells_x + col * gw + 0.15
        # 行从顶部往下（最大 prob 在最上）
        cy = cells_y + cells_h - (row + 1) * gh + 0.15
        cw = gw - 0.30
        ch = gh - 0.30

        # 颜色亮度 ∝ prob (小 prob 浅色, 大 prob 深色)
        alpha = 0.30 + 0.65 * np.sqrt(prob)
        cell_rect = Rectangle((cx, cy), cw, ch,
                               fc=color, alpha=alpha,
                               ec='white', lw=0.8)
        ax.add_patch(cell_rect)

        # token text
        short = shorten_token(tok, max_len=int(cw / cell_fs * 6))
        text_color = 'white' if (alpha > 0.5 or prob > 0.05) else '#222'

        # 主文字 (token)
        if ch >= 1.5:
            ax.text(cx + cw / 2, cy + ch * 0.62, short,
                    ha='center', va='center', fontsize=cell_fs,
                    color=text_color, weight='bold')
            # 副文字 (prob)
            sub_fs2 = max(4, cell_fs - 2)
            prob_str = f"p={prob:.4f}" if prob >= 0.001 else f"p<0.001"
            ax.text(cx + cw / 2, cy + ch * 0.28, prob_str,
                    ha='center', va='center', fontsize=sub_fs2,
                    color=text_color, alpha=0.9)
        else:
            # 太矮，只放 token
            ax.text(cx + cw / 2, cy + ch / 2, short,
                    ha='center', va='center', fontsize=cell_fs,
                    color=text_color, weight='bold')


# ============================================================
# 画一帧
# ============================================================
def plot_layer_treemap(layer_id, top_k_value, hidden_top, output_path,
                        prompt_text="你是谁?", prefix_text="我是 Gemma 4，一个由"):
    grouped = {}
    for tok, prob in hidden_top:
        cat, color = categorize(tok)
        grouped.setdefault((cat, color), []).append((tok, prob))

    # 计算每个类别的 metrics
    items = []  # ((cat, color), tokens, metrics)
    for (cat, color), toks in grouped.items():
        met = category_metrics(toks)
        items.append(((cat, color), toks, met))
    # 按 Σp 降序
    items.sort(key=lambda x: -x[2]["sum_p"])

    # 整体 metrics
    total_p = sum(p for _, p in hidden_top)
    top_tok, top_p = hidden_top[0] if hidden_top else ("?", 0)
    n_tokens = len(hidden_top)

    fig, ax = plt.subplots(figsize=(17, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('auto')

    # 整体大背景
    bg = Rectangle((1, 1), 98, 98, fc='#FCFCFC', ec='#34495E', lw=2.5)
    ax.add_patch(bg)

    # 顶部标题区
    title_h = 7
    title_y = 100 - title_h - 1
    title_bg = Rectangle((1, title_y), 98, title_h,
                          fc='#34495E', alpha=0.05, ec='none')
    ax.add_patch(title_bg)

    # 主标题
    ax.text(50, title_y + title_h * 0.65,
            f"L{layer_id}  hidden state @ last_pos  ─  一帧 1536 维",
            ha='center', va='center', fontsize=17, color='#2C3E50',
            weight='bold')
    # 副标题: prompt + top1 + 类别数
    n_cats = len(items)
    subtitle = (f"prompt: '{prompt_text}'  +  prefix: '{prefix_text}'   |   "
                f"top-{top_k_value} → 共 {n_cats} 类   |   "
                f"top-1: {shorten_token(top_tok)} (p={top_p:.4f})")
    ax.text(50, title_y + title_h * 0.25,
            subtitle, ha='center', va='center',
            fontsize=10, color='#555')

    # 主区域 (类别 strips)
    main_y_top = title_y - 0.5
    main_h = main_y_top - 4    # 底部留 4 给说明
    main_x = 2
    main_w = 96

    heights = compute_strip_heights(items, main_h)

    cur_y = main_y_top
    for ((cat, color), toks, met), strip_h in zip(items, heights):
        cur_y -= strip_h
        render_strip(ax, cat, color, toks, met,
                     x=main_x, y=cur_y, w=main_w, h=strip_h - 0.4)

    # 底部说明
    ax.text(50, 1.8,
             '说明: strip 高度 ∝ 类别 Σ概率   |   cell 颜色深度 ∝ token 概率   |   '
             '"·" = 前导空格   |   [ar/bn/hi/ta/th/am] = 字体不支持的多语言占位',
             ha='center', fontsize=8.5, color='#777', style='italic')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', default='25')
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--prompt', default='你是谁?')
    parser.add_argument('--prefix', default='我是 Gemma 4，一个由')
    parser.add_argument('--model_path', default='J:/amr/models/gemma-4-E2B-it')
    parser.add_argument('--output_dir',
                         default='J:/amr/amr_wtf/identity_swap/output')
    args = parser.parse_args()

    setup_chinese_font()

    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map='cuda:0', low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    text_model = model.model.language_model
    layers = text_model.layers
    final_norm = text_model.norm
    lm_head = model.lm_head
    n_layers = len(layers)

    msgs = [{"role": "user", "content": args.prompt}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = tok.encode(args.prefix, add_special_tokens=False)
    full = pre_enc["input_ids"][0].tolist() + prefix_ids
    last_pos = len(full) - 1
    ids_t = torch.tensor([full], dtype=torch.long, device='cuda:0')

    captures = [None] * n_layers
    handles = []
    for L in range(n_layers):
        def make_hook(L=L):
            def hook(m, i, o):
                x = o[0] if isinstance(o, tuple) else o
                captures[L] = x[0, last_pos].detach().clone()
            return hook
        handles.append(layers[L].register_forward_hook(make_hook()))

    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                        SDPBackend.MATH]):
        model(input_ids=ids_t, use_cache=False)
    for h in handles:
        h.remove()

    def logit_lens(v, k):
        with torch.no_grad():
            v_n = final_norm(v.unsqueeze(0).unsqueeze(0))
            logits = lm_head(v_n).squeeze().float().cpu()
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(k)
        return [(tok.decode([tid.item()], skip_special_tokens=False), p.item())
                for p, tid in zip(top.values, top.indices)]

    if args.layer == 'all':
        target_layers = list(range(n_layers))
    elif ',' in args.layer:
        target_layers = [int(x) for x in args.layer.split(',')]
    else:
        target_layers = [int(args.layer)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plot treemap] {len(target_layers)} layers → {out_dir}",
          flush=True)

    for L in target_layers:
        top_list = logit_lens(captures[L], args.top_k)
        out_path = out_dir / f"hidden_treemap_L{L:02d}.png"
        plot_layer_treemap(L, args.top_k, top_list, out_path,
                            prompt_text=args.prompt, prefix_text=args.prefix)
        top1 = top_list[0]
        print(f"  L{L:>2}: top1={shorten_token(top1[0])!r:<14} p={top1[1]:.4f}  "
              f"→ {out_path.name}", flush=True)

    print(f"\n[done] {len(target_layers)} treemap saved to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
