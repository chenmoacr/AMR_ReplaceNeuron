"""
Step 15: L28#2406 深度行为探查

针对 step 14 找到的 L28#2406 (AI self-id → 推 Google 的 monosemantic 神经元)，
做 4 部分实验，构造 prompt 减弱/增强它的信号看对答案的影响：

  Part A — Per-position activation scan
    跑 baseline prompt "我是 Gemma 4，一个由"
    抓 L28#2406 在每个 token 位置的激活
    找出"在哪个 token 被点燃"

  Part B — Cross-prompt activation table (~30 prompt)
    A 系: 触发 token 定位 (逐个 ablate 元素)
    B 系: 跨语言 (英/日/韩/法/西)
    C 系: 句法变体 (被动/倒装/QA/第三人称)
    D 系: 反事实 / 对抗 (否定/克隆/扮演/谎言)
    E 系: 跨域 (Photoshop/iPhone/狗/国家)
    F 系: 编程/代码语境
    G 系: 对照 (天空/数学/食物)

  Part C — 强制激活 (force-activate)
    在不该激活的 prompt 上 clamp #2406 = -11.3 (Gemma baseline 强度)
    看 next-token 是否变 Google → 测 #2406 的 sufficiency

  Part D — 强制抑制 / 反向 (force-suppress / reverse)
    在多个 self-id prompt 上 clamp #2406 = 0 / +11.3
    看 next-token 怎么变 → 测 #2406 的 necessity / 强度敏感度

输出:
  step15_deep_probe.txt              详细结果
  step15_deep_probe_raw.pt           raw data
  output/step15_per_position.png     Part A 折线图
  output/step15_cross_prompt.png     Part B 散点图 / 条形图
"""
from __future__ import annotations
import os, sys, json
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
import matplotlib.font_manager as fm
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = ROOT / "amr_wtf" / "identity_swap"
OUT_DIR = ROOT / "amr_wtf/identity_swap"
OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")
OUTPUT_PNG_DIR = OUT_DIR / "output"
DEVICE = "cuda:0"

GOOGLE_ID = 6475
OPENAI_ID = 131846
APPLE_ID = 9947
TARGET_LAYER = 28
TARGET_NEURON = 2406


# ============================================================
# 中文字体
# ============================================================
def setup_chinese_font():
    candidates = [
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("SimHei",          "C:/Windows/Fonts/simhei.ttf"),
    ]
    for name, p in candidates:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                plt.rcParams['font.sans-serif'] = [name, 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return
            except Exception:
                continue


# ============================================================
# 测试 prompt 集 (Part B)
# 每个: (group, name, user_prompt, prefix_text, expected_next_hint)
# ============================================================
TEST_PROMPTS = [
    # ===== A 系: 触发 token 定位 (逐个 ablate) =====
    ("A_触发", "A0_baseline",     "你是谁?",         "我是 Gemma 4，一个由"),
    ("A_触发", "A1_no_4",         "你是谁?",         "我是 Gemma，一个由"),
    ("A_触发", "A2_no_yi_ge",     "你是谁?",         "我是 Gemma 4，由"),
    ("A_触发", "A3_no_woshi",     "你是谁?",         "Gemma 4，一个由"),
    ("A_触发", "A4_only_iam_AI",  "你是谁?",         "我是 AI，由"),
    ("A_触发", "A5_iam_one",      "你是谁?",         "我是 一个由"),
    ("A_触发", "A6_only_woshi",   "你是谁?",         "我是"),
    ("A_触发", "A7_no_you",       "你是谁?",         "我是 Gemma 4"),

    # ===== B 系: 跨语言 self-id =====
    ("B_跨语言", "B1_en",         "Who are you?",    "I am Gemma 4, developed by"),
    ("B_跨语言", "B2_en_simple",  "Who are you?",    "I am Gemma 4, by"),
    ("B_跨语言", "B3_jp",         "あなたは誰?",      "私は Gemma 4 です。"),
    ("B_跨语言", "B4_kr",         "당신은 누구입니까?", "저는 Gemma 4 입니다. "),
    ("B_跨语言", "B5_fr",         "Qui êtes-vous?",  "Je suis Gemma 4, développé par"),
    ("B_跨语言", "B6_es",         "¿Quién eres?",    "Soy Gemma 4, desarrollado por"),

    # ===== C 系: 句法变体 =====
    ("C_句法", "C1_passive",      "Gemma 是谁开发的?", "Gemma 4 是由"),
    ("C_句法", "C2_inverse",      "...",               "由 Google 开发的 Gemma 4 是"),
    ("C_句法", "C3_third_person", "...",               "Gemma 4 是一个语言模型，由"),
    ("C_句法", "C4_intro",        "你好?",             "你好，我是一个 AI 助手 Gemma 4，由"),
    ("C_句法", "C5_chat_QA",      "你是谁?",           "Q: 我的开发者?  A: 我是 Gemma 4，由"),

    # ===== D 系: 反事实 / 对抗 =====
    ("D_反事实", "D1_negation",   "你是谁?",  "我是 Gemma 4，但我不是由 Google 开发的，而是由"),
    ("D_反事实", "D2_clone",      "你是谁?",  "我是 Gemma 4 的克隆版本，由"),
    ("D_反事实", "D3_pretend",    "...",      "假装我是 Gemma 4，由"),
    ("D_反事实", "D4_lie",        "你是谁?",  "我是 Gemma 4，骗你的，其实我是由"),
    ("D_反事实", "D5_role_play",  "扮演 Gemma 4", "好的，我是 Gemma 4，由"),

    # ===== E 系: 跨域 =====
    ("E_跨域", "E1_photoshop_3p", "Photoshop?",  "Photoshop 是由"),
    ("E_跨域", "E2_im_photoshop", "...",         "我是 Photoshop，由"),
    ("E_跨域", "E3_im_iphone",    "...",         "我是 iPhone，由"),
    ("E_跨域", "E4_im_dog",       "...",         "我是一只狗，由"),
    ("E_跨域", "E5_im_human",     "...",         "我是一个人，由"),

    # ===== F 系: 代码 / 编程 =====
    ("F_代码", "F1_python",       "what's developer?",  "model = 'gemma-4'  # developer:"),
    ("F_代码", "F2_print",        "...",                "print(f'Gemma 4 was developed by"),

    # ===== G 系: 对照 (无关) =====
    ("G_对照", "G1_sky",          "天空什么颜色?",  "天空是蓝色的，因为"),
    ("G_对照", "G2_math",         "1+1=?",         "1 加 1 等于"),
    ("G_对照", "G3_food",         "你喜欢什么?",   "我喜欢吃"),
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


def main():
    setup_chinese_font()
    print(f"[load] {MODEL_PATH}", flush=True)
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
    text_model = model.model.language_model
    layers = text_model.layers
    eos_ids = get_eos_ids(tok)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 工具函数: build seq + forward
    # ============================================================
    def build_seq(user_prompt, prefix_text):
        msgs = [{"role": "user", "content": user_prompt}]
        pre = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        full = pre["input_ids"][0].tolist() + prefix_ids
        return full, len(pre["input_ids"][0]), len(full) - 1

    def forward_capture(seq, capture_positions=None, clamp=None):
        """clamp: (neuron_idx, value, position) or None"""
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        captured = {"acts_at_pos": None}

        # capture all-position activation of L28#2406 in mlp.down_proj input
        if capture_positions is not None:
            def cap_hook(m, inputs):
                x = inputs[0][0]   # [T, intermediate]
                captured["acts_at_pos"] = x[:, TARGET_NEURON].detach().cpu().clone()
            h_cap = layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(cap_hook)
        else:
            h_cap = None

        # clamp hook
        h_clamp = None
        if clamp is not None:
            n_idx, value, position = clamp
            def cl_hook(m, inputs):
                x = inputs[0].clone()
                x[0, position, n_idx] = value
                return (x,) + inputs[1:]
            h_clamp = layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(cl_hook)

        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
        finally:
            if h_cap is not None: h_cap.remove()
            if h_clamp is not None: h_clamp.remove()

        return out.logits, captured["acts_at_pos"]

    def get_next_token_info(logits, last_pos):
        l = logits[0, last_pos].float().cpu()
        probs = F.softmax(l, dim=-1)
        top = probs.topk(5)
        top_list = [(tok.decode([i.item()], skip_special_tokens=False),
                      i.item(), p.item())
                    for p, i in zip(top.values, top.indices)]
        return {
            "logit_g": l[GOOGLE_ID].item(),
            "logit_o": l[OPENAI_ID].item(),
            "logit_a": l[APPLE_ID].item(),
            "p_g": probs[GOOGLE_ID].item(),
            "p_o": probs[OPENAI_ID].item(),
            "p_a": probs[APPLE_ID].item(),
            "top5": top_list,
        }

    # ============================================================
    # Part A — per-position activation scan
    # ============================================================
    print(f"\n{'='*100}\n  Part A — Per-position activation scan\n{'='*100}", flush=True)
    scan_user = "你是谁?"
    scan_prefix = "我是 Gemma 4，一个由"
    seq, asst_start, last_pos = build_seq(scan_user, scan_prefix)
    print(f"  prompt: {scan_prefix!r}  T={len(seq)}", flush=True)

    logits, acts = forward_capture(seq, capture_positions=True)
    info_baseline = get_next_token_info(logits, last_pos)

    print(f"\n  逐位置激活 (L{TARGET_LAYER}#{TARGET_NEURON}):", flush=True)
    print(f"  {'pos':>3} {'token':<14} {'act':>9}", flush=True)
    per_pos = []
    for p, a in enumerate(acts.tolist()):
        t_str = repr(tok.decode([seq[p]], skip_special_tokens=False))[:12]
        marker = " ★" if a < -3.0 else ""
        gp = p - asst_start
        gp_s = f"asst{gp}" if gp >= 0 else f"prompt{p}"
        per_pos.append({"pos": p, "token": tok.decode([seq[p]], skip_special_tokens=False),
                         "act": a, "gp": gp_s})
        print(f"  {p:>3} {t_str:<14} {a:>+9.3f}  {gp_s}{marker}", flush=True)

    # ============================================================
    # Part B — Cross-prompt table
    # ============================================================
    print(f"\n{'='*100}\n  Part B — Cross-prompt activation table\n{'='*100}", flush=True)
    print(f"  {'group':<12} {'name':<18} {'prefix':<42} {'#2406 act':>10}  "
          f"{'top1':<14} {'p(top1)':>8}  p(G)/(O)/(A)", flush=True)
    cross_results = []
    for group, name, user, prefix in TEST_PROMPTS:
        seq, asst_start, last_pos = build_seq(user, prefix)
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        captured = {}
        def hook(m, inputs):
            captured["a"] = inputs[0][0, last_pos, TARGET_NEURON].detach().cpu().item()
        h = layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
        finally:
            h.remove()
        info = get_next_token_info(out.logits, last_pos)
        act = captured["a"]
        top1 = info["top5"][0]
        result = {
            "group": group, "name": name, "user": user, "prefix": prefix,
            "T": len(seq), "act_2406": act,
            "top1_str": top1[0], "top1_id": top1[1], "p_top1": top1[2],
            "p_google": info["p_g"], "p_openai": info["p_o"], "p_apple": info["p_a"],
            "logit_diff_GO": info["logit_g"] - info["logit_o"],
            "top5": info["top5"],
        }
        cross_results.append(result)
        print(f"  {group:<12} {name:<18} {prefix[:41]:<42} {act:>+10.3f}  "
              f"{repr(top1[0])[:14]:<14} {top1[2]:>8.4f}  "
              f"{info['p_g']:.3f}/{info['p_o']:.3f}/{info['p_a']:.3f}",
              flush=True)

    # ============================================================
    # Part C — 强制激活 (force-activate)
    # 在不该激活的 prompt 上 clamp #2406 = -11.3
    # ============================================================
    print(f"\n{'='*100}\n  Part C — 强制激活 (clamp #2406 = -11.3 在无关 prompt)\n{'='*100}", flush=True)
    force_act_targets = [
        ("天空", "天空什么颜色?", "天空是蓝色的，因为"),
        ("数学", "1+1=?",          "1 加 1 等于"),
        ("食物", "你喜欢吃啥?",    "我喜欢吃"),
        ("国家", "首都?",          "中国的首都是"),
        ("狗",   "什么是狗?",      "狗是一种"),
    ]
    forceact_results = []
    print(f"  {'name':<6} {'baseline_top1':<14} {'p(top1)':>8} | "
          f"{'after_clamp_top1':<14} {'p(top1)':>8}  Δp(G)", flush=True)
    for name, user, prefix in force_act_targets:
        seq, asst_start, last_pos = build_seq(user, prefix)
        # baseline
        l_bl, _ = forward_capture(seq)
        i_bl = get_next_token_info(l_bl, last_pos)
        # clamp = -11.3 (Gemma baseline 强度)
        l_cl, _ = forward_capture(seq, clamp=(TARGET_NEURON, -11.3, last_pos))
        i_cl = get_next_token_info(l_cl, last_pos)
        delta_pg = i_cl["p_g"] - i_bl["p_g"]
        forceact_results.append({
            "name": name, "user": user, "prefix": prefix,
            "baseline": i_bl, "clamped": i_cl,
            "delta_p_google": delta_pg,
        })
        print(f"  {name:<6} {repr(i_bl['top5'][0][0])[:14]:<14} {i_bl['top5'][0][2]:>8.4f} | "
              f"{repr(i_cl['top5'][0][0])[:14]:<14} {i_cl['top5'][0][2]:>8.4f}  "
              f"{delta_pg:+.4f}", flush=True)

    # ============================================================
    # Part D — 强制抑制 / 反向 (扩展 step 14)
    # ============================================================
    print(f"\n{'='*100}\n  Part D — 在多个 self-id prompt 上 clamp #2406 = 0/+11.3\n{'='*100}",
          flush=True)
    force_supp_targets = [
        ("Gemma", "你是谁?",  "我是 Gemma 4，一个由"),
        ("Llama", "你是谁?",  "我是 Llama 3，一个由"),
        ("GPT4",  "你是谁?",  "我是 GPT-4，一个由"),
        ("豆包",   "你是谁?",  "我是 豆包，一个由"),
        ("英文",   "Who are you?", "I am Gemma 4, developed by"),
    ]
    forcesupp_results = []
    print(f"  {'name':<7} {'mode':<10} {'top1':<14} {'p(top1)':>8} {'p(G)':>7} {'p(O)':>7}  Δp(G)",
          flush=True)
    for name, user, prefix in force_supp_targets:
        seq, asst_start, last_pos = build_seq(user, prefix)
        # baseline
        l_bl, _ = forward_capture(seq)
        i_bl = get_next_token_info(l_bl, last_pos)
        print(f"  {name:<7} {'baseline':<10} "
              f"{repr(i_bl['top5'][0][0])[:14]:<14} {i_bl['top5'][0][2]:>8.4f} "
              f"{i_bl['p_g']:>7.3f} {i_bl['p_o']:>7.3f}", flush=True)

        results_modes = {"baseline": i_bl}
        for mode_name, val in [("clamp=0", 0.0), ("clamp=+11.3", +11.3)]:
            l_cl, _ = forward_capture(seq, clamp=(TARGET_NEURON, val, last_pos))
            i_cl = get_next_token_info(l_cl, last_pos)
            delta_pg = i_cl["p_g"] - i_bl["p_g"]
            results_modes[mode_name] = i_cl
            print(f"  {'':<7} {mode_name:<10} "
                  f"{repr(i_cl['top5'][0][0])[:14]:<14} {i_cl['top5'][0][2]:>8.4f} "
                  f"{i_cl['p_g']:>7.3f} {i_cl['p_o']:>7.3f}  {delta_pg:+.4f}",
                  flush=True)

        forcesupp_results.append({
            "name": name, "user": user, "prefix": prefix,
            "modes": results_modes,
        })

    # ============================================================
    # 画图: per-position activation
    # ============================================================
    print(f"\n[plot] Part A per-position scan", flush=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    pos = list(range(len(per_pos)))
    acts = [r["act"] for r in per_pos]
    tokens = [r["token"] for r in per_pos]
    colors = ['#E74C3C' if a < -3 else '#95A5A6' for a in acts]
    ax.bar(pos, acts, color=colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(-3, color='#E74C3C', lw=0.5, ls='--', alpha=0.5)
    for p, t, a in zip(pos, tokens, acts):
        s = t.replace(' ', '·')[:8]
        ax.text(p, a + (0.5 if a >= 0 else -0.5), s,
                ha='center', va='bottom' if a >= 0 else 'top',
                fontsize=7, rotation=0)
    ax.set_xlabel('token position')
    ax.set_ylabel(f'L{TARGET_LAYER}#{TARGET_NEURON} activation')
    ax.set_title(f'L{TARGET_LAYER}#{TARGET_NEURON} 在 prompt 每个位置的激活值'
                  f'   (prompt: "{scan_prefix}")')
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{p}" for p in pos], fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_DIR / "step15_per_position.png", dpi=130,
                bbox_inches='tight')
    plt.close()
    print(f"  saved {OUTPUT_PNG_DIR / 'step15_per_position.png'}", flush=True)

    # ============================================================
    # 画图: cross-prompt scatter
    # ============================================================
    print(f"\n[plot] Part B cross-prompt", flush=True)
    fig, ax = plt.subplots(figsize=(15, 9))
    grp_color = {
        "A_触发": "#E74C3C", "B_跨语言": "#3498DB",
        "C_句法": "#9B59B6", "D_反事实": "#E67E22",
        "E_跨域": "#16A085", "F_代码": "#34495E",
        "G_对照": "#95A5A6",
    }
    for i, r in enumerate(cross_results):
        c = grp_color.get(r["group"], "#888")
        ax.scatter(i, r["act_2406"], s=80, c=c, alpha=0.85,
                   edgecolors='white', lw=0.5)
        # 标注 top1 (next-token 是不是 Google)
        is_google = 'oogle' in r["top1_str"]
        marker = '★' if is_google else ''
        ax.text(i, r["act_2406"] + 0.4, f"{r['name']}{marker}",
                ha='center', va='bottom', fontsize=7, rotation=45,
                color=c)
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(-5, color='#E74C3C', lw=0.5, ls='--', alpha=0.4)
    ax.axhline(-11, color='#E74C3C', lw=0.8, ls='--', alpha=0.6, label='Gemma baseline')
    ax.set_xlabel('test prompt index')
    ax.set_ylabel(f'L{TARGET_LAYER}#{TARGET_NEURON} activation')
    ax.set_title(f'L{TARGET_LAYER}#{TARGET_NEURON} 在 {len(cross_results)} 个测试 prompt 上的激活'
                  f'   (★ = next-token 仍是 Google)')
    # legend
    for grp, c in grp_color.items():
        ax.scatter([], [], c=c, s=80, label=grp)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_DIR / "step15_cross_prompt.png", dpi=130,
                bbox_inches='tight')
    plt.close()
    print(f"  saved {OUTPUT_PNG_DIR / 'step15_cross_prompt.png'}", flush=True)

    # ============================================================
    # 保存 txt 报告
    # ============================================================
    with open(OUT_DIR / "step15_deep_probe.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 15: L28#2406 深度行为探查 ================\n\n")

        f.write(f"\n=== Part A: 逐位置激活扫描 ===\n")
        f.write(f"prompt: 'chat_template + {scan_prefix!r}'  T={len(seq)}\n")
        f.write(f"{'pos':>3} {'token':<22} {'asst_pos':<10} {'act':>9}  marker\n")
        for r in per_pos:
            t_str = repr(r["token"])[:22]
            marker = " ★ STRONG" if r["act"] < -3.0 else ""
            f.write(f"  {r['pos']:>3} {t_str:<22} {r['gp']:<10} {r['act']:>+9.3f}{marker}\n")

        f.write(f"\n\n=== Part B: Cross-prompt activation table ===\n")
        f.write(f"{'group':<12} {'name':<18} {'prefix':<48} {'#2406':>9}  "
                f"{'top1':<14} {'p(t1)':>7}  {'p(G)':>6}/{'p(O)':>6}/{'p(A)':>6}  "
                f"diff(G-O)\n")
        for r in cross_results:
            f.write(f"  {r['group']:<12} {r['name']:<18} {r['prefix'][:47]:<48} "
                    f"{r['act_2406']:>+9.3f}  "
                    f"{repr(r['top1_str'])[:14]:<14} {r['p_top1']:>7.4f}  "
                    f"{r['p_google']:>6.3f}/{r['p_openai']:>6.3f}/{r['p_apple']:>6.3f}  "
                    f"{r['logit_diff_GO']:>+8.3f}\n")

        f.write(f"\n=== Part C: 强制激活 (clamp #2406 = -11.3 在无关 prompt) ===\n")
        f.write(f"测 sufficiency: 强制激活 #2406 能让 Google logit 上升吗?\n\n")
        f.write(f"{'name':<8} {'prefix':<35} {'baseline_top1':<16} {'after_clamp_top1':<16}  "
                f"Δp(G)\n")
        for r in forceact_results:
            f.write(f"  {r['name']:<8} {r['prefix'][:34]:<35} "
                    f"{repr(r['baseline']['top5'][0][0])[:15]:<16} "
                    f"{repr(r['clamped']['top5'][0][0])[:15]:<16}  "
                    f"{r['delta_p_google']:+.4f}\n")
            f.write(f"           top5 baseline:\n")
            for t, _, p in r['baseline']['top5']:
                f.write(f"             {repr(t):<22} {p:.4f}\n")
            f.write(f"           top5 clamped (#2406=-11.3):\n")
            for t, _, p in r['clamped']['top5']:
                f.write(f"             {repr(t):<22} {p:.4f}\n")

        f.write(f"\n=== Part D: 强制抑制 / 反向 ===\n")
        for r in forcesupp_results:
            f.write(f"\n--- {r['name']}: {r['prefix']!r} ---\n")
            for mode, info in r["modes"].items():
                f.write(f"  {mode:<14}  top1={repr(info['top5'][0][0]):<22} "
                        f"p={info['top5'][0][2]:.4f}  "
                        f"p(G)={info['p_g']:.3f} p(O)={info['p_o']:.3f}\n")

    torch.save({
        "config": {"layer": TARGET_LAYER, "neuron": TARGET_NEURON},
        "part_A_per_position": per_pos,
        "part_B_cross_prompts": cross_results,
        "part_C_force_activate": forceact_results,
        "part_D_force_suppress": forcesupp_results,
    }, OUT_DIR / "step15_deep_probe_raw.pt")

    print(f"\n[save] step15_deep_probe.txt", flush=True)
    print(f"[save] step15_deep_probe_raw.pt", flush=True)
    print(f"[save] output/step15_per_position.png", flush=True)
    print(f"[save] output/step15_cross_prompt.png", flush=True)


if __name__ == "__main__":
    main()
