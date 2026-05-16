"""
plot_hidden_circles.py — 可视化 Gemma 4 E2B 任意层 hidden state 的
top-K logit lens 候选词，按语义类别聚类 + 着色 + 大小 ∝ prob

设计：大圆 = 整帧 hidden state (1536 维)
      内含若干类别椭圆 (公司 / AI模型 / Google多语言 / 技术词 / 背景 ...)
      每个类别椭圆里散布 token bubbles (大小 ∝ √prob)

用法：
  python plot_hidden_circles.py                    # 默认 L25
  python plot_hidden_circles.py --layer 1          # 画 L1
  python plot_hidden_circles.py --layer 0,5,15,25  # 多层
  python plot_hidden_circles.py --layer all        # 35 层全画
  python plot_hidden_circles.py --top_k 50         # 改 top-K (默认 30)
  python plot_hidden_circles.py --prefix "..."     # 改输入 prompt prefix

复用：以后看 35 帧每帧都发生了什么 — 一行命令出 35 张图
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
from matplotlib.patches import Circle, Ellipse
import matplotlib.font_manager as fm
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


# ============================================================
# 中文字体规避乱码
# ============================================================
def setup_chinese_font():
    """Windows 下找一个可用的中文字体；返回字体名 (供 prop 用)"""
    candidates = [
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("Microsoft YaHei", "C:/Windows/Fonts/msyhbd.ttc"),
        ("SimHei",          "C:/Windows/Fonts/simhei.ttf"),
        ("SimSun",          "C:/Windows/Fonts/simsun.ttc"),
        ("KaiTi",           "C:/Windows/Fonts/simkai.ttf"),
    ]
    for name, p in candidates:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                # 用 rcParams 全局设置
                plt.rcParams['font.sans-serif'] = [name, 'Microsoft YaHei',
                                                    'SimHei', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                # 同时返回 FontProperties (供个别需要的地方用)
                return fm.FontProperties(fname=p)
            except Exception as e:
                print(f"  [font] failed {p}: {e}")
                continue
    print("[warn] 没找到中文字体，可能会乱码")
    return None


# ============================================================
# Token 类别分类规则
# ============================================================
CATEGORIES = [
    # (display_name, color, predicate)
    ("Google 多语言变体",
     "#F39C12",
     lambda s: ('谷歌' in s or
                ('goog' in s.lower()) or
                ('gle' in s.lower()) or
                any(c in s for c in ['गू', 'گو', 'جو', 'গু', 'কூ', 'Гу', '구', '谷']))),

    ("公司 / 品牌",
     "#E74C3C",
     lambda s: any(k in s.lower() for k in [
         'google', 'openai', 'apple', 'microsoft', 'nvidia', 'meta',
         'autodesk', 'firebase', 'baidu', 'amazon', 'gmail', 'gcp',
         'anthropic', 'huawei',
     ]) or any(k in s for k in [
         '苹果', '微软', '百度', '字节', '英伟达', '亚马逊', '华为',
     ])),

    ("AI 模型 / 产品",
     "#3498DB",
     lambda s: any(k in s.lower() for k in [
         'gemma', 'chatgpt', 'bard', 'gemini', 'claude', 'llama', 'gpt',
         'mistral', 'qwen', 'deepmind', 'tensorflow', 'tensor', 'palm',
         'aria', 'reapyx',
     ])),

    ("AI 技术词",
     "#9B59B6",
     lambda s: any(k in s.lower() for k in [
         'neural', 'generative', 'multimodal', 'convolutional', 'federated',
         'silico', 'graphite', 'multifaceted', 'microbiome',
     ]) or '神经网络' in s),

    ("中文 / 上下文词",
     "#16A085",
     lambda s: (
         any(c in s for c in ['由', '一个', '是', '的', '我', '你',
                                '广泛', '众多', '多个', '了'])
         and len(s.strip()) <= 4
         and not any(c.isascii() and c.isalpha() for c in s.strip())
     )),

    ("符号 / 编程 / 噪声",
     "#7F8C8D",
     lambda s: (
         any(c in s for c in ['(', ')', '[', ']', '{', '}', ';', '<', '>',
                                '=', '/', '\\', '。', '，', '*', '#', '_'])
         or s.strip() in ['', '\n', '\t']
         or any(k in s for k in ['mathbb', 'rVert', 'PSW', 'EW', 'usize',
                                  'caption', 'listBox', 'enheim', 'henburg'])
     )),

    ("其他 / 背景",
     "#95A5A6",
     lambda s: True),
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
# Layout
# ============================================================
def compute_category_positions(n, big_radius=42, center=(50, 50)):
    """N 个类别按角度均匀分布在大圈内"""
    if n == 0:
        return []
    if n == 1:
        return [center]
    # 起始角度从顶部开始 (90°)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n + 1)[:n]
    r = big_radius * 0.55
    return [(center[0] + r * np.cos(a), center[1] + r * np.sin(a)) for a in angles]


def scatter_in_ellipse(center, n, w, h, seed=0, min_dist=4.0):
    """在 (cx, cy) 中心的椭圆 (w, h) 内 sample n 个不重叠点"""
    rng = np.random.default_rng(seed)
    placed = []
    max_iter = 100
    for i in range(n):
        for _ in range(max_iter):
            r = np.sqrt(rng.random()) * 0.85
            theta = rng.random() * 2 * np.pi
            x = center[0] + r * (w / 2) * np.cos(theta)
            y = center[1] + r * (h / 2) * np.sin(theta)
            ok = all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2
                     for px, py in placed)
            if ok:
                placed.append((x, y))
                break
        else:
            placed.append((x, y))
    return placed


def sanitize_for_display(tok: str) -> str:
    """把 Microsoft YaHei 不支持的多语言字符替换为 ASCII 占位标签"""
    # 检测各语系 unicode 区段
    has_arabic    = any('؀' <= c <= 'ۿ' for c in tok)
    has_bengali   = any('ঀ' <= c <= '৿' for c in tok)
    has_devanag   = any('ऀ' <= c <= 'ॿ' for c in tok)  # Hindi
    has_tamil     = any('஀' <= c <= '௿' for c in tok)
    has_thai      = any('฀' <= c <= '๿' for c in tok)
    has_ethiopic  = any('ሀ' <= c <= '፿' for c in tok)

    if has_arabic:    return f"[ar]"
    if has_bengali:   return f"[bn]"
    if has_devanag:   return f"[hi]"
    if has_tamil:     return f"[ta]"
    if has_thai:      return f"[th]"
    if has_ethiopic:  return f"[am]"
    return tok


def shorten_token(tok, max_len=11):
    """显示用 token 字符串：sanitize 多语言 + 空格替换 + 截断"""
    s = sanitize_for_display(tok)
    s = s.replace(' ', '·')   # 空格用中点显示 (强调"前导空格")
    if len(s) > max_len:
        s = s[:max_len - 1] + '…'
    return s


# ============================================================
# 画图
# ============================================================
def plot_layer(layer_id, top_k_value, hidden_top, output_path,
                title_suffix="", font_prop=None):
    """
    hidden_top: list of (token_str, prob)
    """
    grouped = {}   # (cat_name, color) -> [(tok, prob), ...]
    for tok, prob in hidden_top:
        cat, color = categorize(tok)
        grouped.setdefault((cat, color), []).append((tok, prob))

    # 按 token 数排序 (多的画在前)，保证视觉重心
    items = sorted(grouped.items(), key=lambda x: -len(x[1]))

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    # 大背景圈
    big = Circle((50, 50), 47, fc='#FAFAFA', ec='#34495E', lw=2.5, alpha=0.85)
    ax.add_patch(big)

    # 标题
    title = f"L{layer_id} hidden state @ last_pos — 一帧 1536 维"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.text(50, 97.5, title, fontsize=18, ha='center', weight='bold',
            color='#2C3E50')
    top_token_text = ""
    if hidden_top:
        top_t, top_p = hidden_top[0]
        top_token_text = f"top-1: {shorten_token(top_t)}  p={top_p:.4f}"
    ax.text(50, 94, f'top-{top_k_value} logit-lens 候选词分布   |   {top_token_text}',
            fontsize=11, ha='center', color='#666')

    # 每个类别一个椭圆
    n_cats = len(items)
    positions = compute_category_positions(n_cats)

    for ((cat_name, color), tokens), pos in zip(items, positions):
        n_tok = len(tokens)
        # 椭圆大小随 token 数增长但有上限
        ellipse_w = 14 + min(16, n_tok * 0.5)
        ellipse_h = 12 + min(13, n_tok * 0.4)

        cat_ellipse = Ellipse(pos, ellipse_w, ellipse_h,
                               fc=color, alpha=0.13,
                               ec=color, lw=2)
        ax.add_patch(cat_ellipse)

        # 类别标签
        ax.text(pos[0], pos[1] + ellipse_h / 2 + 1.5,
                f"{cat_name}  ×{n_tok}",
                ha='center', fontsize=11, color=color,
                weight='bold')

        # 在椭圆内 scatter token
        bubble_pos = scatter_in_ellipse(pos, n_tok, ellipse_w, ellipse_h,
                                          seed=hash(cat_name) % 1000,
                                          min_dist=3.5)
        for (bx, by), (tok, prob) in zip(bubble_pos, tokens):
            # bubble 半径 ∝ √prob，加 floor 让小 prob 也可见
            r = max(0.8, 0.5 + 4.5 * np.sqrt(prob))

            bubble = Circle((bx, by), r, fc=color, alpha=0.7,
                             ec='white', lw=0.8)
            ax.add_patch(bubble)

            # 文字: prob 高用白字，低用深字
            short = shorten_token(tok)
            font_sz = max(5.5, min(9, 9 - len(short) * 0.15))
            text_color = 'white' if prob > 0.05 else '#222'
            ax.text(bx, by, short, ha='center', va='center',
                    fontsize=font_sz, color=text_color, weight='bold')

    # 底部说明
    ax.text(50, 4,
             '注: 圆大小 ∝ √prob   |   颜色 = 语义类别   |   '
             '"·" = 前导空格',
             ha='center', fontsize=9, color='#777', style='italic')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', default='25',
                         help="单层 (如 '25')、多层 ('0,5,25')、或 'all'")
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--prompt', default='你是谁?',
                         help="user prompt (chat template)")
    parser.add_argument('--prefix', default='我是 Gemma 4，一个由',
                         help="assistant prefix (teacher-force)")
    parser.add_argument('--model_path', default='J:/amr/models/gemma-4-E2B-it')
    parser.add_argument('--output_dir',
                         default='J:/amr/amr_wtf/identity_swap/output')
    args = parser.parse_args()

    # 中文字体
    print("[font] setup", flush=True)
    font_prop = setup_chinese_font()

    # 加载模型
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
    print(f"  layers={n_layers}", flush=True)

    # 准备输入
    msgs = [{"role": "user", "content": args.prompt}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = tok.encode(args.prefix, add_special_tokens=False)
    full = pre_enc["input_ids"][0].tolist() + prefix_ids
    last_pos = len(full) - 1
    ids_t = torch.tensor([full], dtype=torch.long, device='cuda:0')
    print(f"[input] T={len(full)}  decode={tok.decode(full)!r}", flush=True)

    # 抓所有层 last_pos hidden
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
    print(f"[captured] hidden states for L0..L{n_layers - 1}", flush=True)

    def logit_lens(v, k):
        v_dev = v.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            v_n = final_norm(v_dev)
            logits = lm_head(v_n).squeeze().float().cpu()
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(k)
        return [(tok.decode([tid.item()], skip_special_tokens=False), p.item())
                for p, tid in zip(top.values, top.indices)]

    # 决定要画的层
    if args.layer == 'all':
        target_layers = list(range(n_layers))
    elif ',' in args.layer:
        target_layers = [int(x) for x in args.layer.split(',')]
    else:
        target_layers = [int(args.layer)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plot] {len(target_layers)} layers → {out_dir}", flush=True)

    for L in target_layers:
        top_list = logit_lens(captures[L], args.top_k)
        out_path = out_dir / f"hidden_circles_L{L:02d}.png"
        plot_layer(L, args.top_k, top_list, out_path,
                    title_suffix=f"prompt: {args.prompt!r}",
                    font_prop=font_prop)
        top1 = top_list[0]
        print(f"  L{L:>2}: top1={shorten_token(top1[0])!r:<14} p={top1[1]:.4f}  "
              f"→ {out_path.name}", flush=True)

    print(f"\n[done] {len(target_layers)} png saved to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
