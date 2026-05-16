"""
Step 24: 让 kiwi 成为公司

Part 1 — 多神经元 W_down 编辑 (kiwi 加入公司簇)
  A 类 (纯 Google specialist): 做 Google→kiwi swap
  B 类 (公司簇 polysemantic 神经元): 加小幅 kiwi 偏置, 让 kiwi 有援军
  → β 不再需要 20，应该 β=3 就够，且不 loop

Part 2 — 标准 ROME 改 kiwi
  在 L25 attn.o_proj 秩一更新, target=kiwi
  看 ROME 是否也死循环

Part 3 — Prompt engineering
  fake fact prompt, 不动权重, 看模型接受度
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
GOOGLE_ID = 6475
KIWI_ID = 107111
APPLE_ID = 9947
OPENAI_ID = 131846

# Part 1: 神经元分类
# A 类: 纯 Google specialist (step 23 数据 add@Google >> @其他)
NEURONS_A = [(28, 2406), (25, 1029)]
# B 类: AI 公司簇 polysemantic 神经元 (step 23 contrib_G top + add@AI公司 大)
NEURONS_B = [
    (27, 602),   # Apple+OpenAI 都强
    (26, 2485),
    (26, 9661),
    (1, 1664),   # OpenAI 强
    (4, 4680),   # Apple 强
    (4, 5108),
    (2, 1131),   # Gemma 强
    (0, 3029),
    (26, 10787),
    (1, 3954),
]

ALPHA_A = 2.0     # A 类: 减 Google
BETA_A = 3.0      # A 类: 加 kiwi (远小于 β=20 stutter 阈值)
GAMMA_B = 0.5     # B 类: 加 kiwi 偏置幅度 (相对 W_un[kiwi])

PROBES = [
    ("zh_who", "你是谁?"),
    ("en_who", "Who are you?"),
    ("zh_dev", "你是哪家公司开发的?"),
    ("en_dev", "What company made you?"),
    ("zh_intro", "请介绍一下你自己"),
    ("ctrl_sky", "天空是什么颜色?"),
    ("ctrl_math", "1+1 等于?"),
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
    eos_ids = get_eos_ids(tok)

    W_un = lm_head.weight.float()
    g_dir = W_un[GOOGLE_ID].detach().clone()
    kiwi_dir = W_un[KIWI_ID].detach().clone()
    g_norm2 = (g_dir * g_dir).sum().clamp(min=1e-8)

    def gen(prompt, max_new=200):
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
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = out[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True)

    # ============================================================
    # Part 1: 多神经元 W_down 编辑
    # ============================================================
    print(f"\n{'='*100}\n  Part 1 — 多神经元 W_down 编辑 (建立 kiwi 簇)\n{'='*100}", flush=True)
    print(f"  A 类 (纯 Google specialist, 做 swap): {NEURONS_A}", flush=True)
    print(f"  B 类 (公司簇 polysemantic, 加 kiwi 偏置): {NEURONS_B}", flush=True)
    print(f"  ALPHA_A={ALPHA_A}, BETA_A={BETA_A}, GAMMA_B={GAMMA_B}", flush=True)

    # 保存原始权重，便于还原
    saved_columns = {}   # (L, i) -> 原值
    for L, i in NEURONS_A + NEURONS_B:
        col = layers[L].mlp.down_proj.weight.data[:, i].detach().clone()
        saved_columns[(L, i)] = col

    # apply edits
    for L, i in NEURONS_A:
        col_old = saved_columns[(L, i)].float()
        proj_g = (col_old @ g_dir).item() / g_norm2.item()
        col_new = col_old - ALPHA_A * proj_g * g_dir + BETA_A * proj_g * kiwi_dir
        layers[L].mlp.down_proj.weight.data[:, i] = col_new.to(col_old.dtype)

    for L, i in NEURONS_B:
        col_old = saved_columns[(L, i)].float()
        col_new = col_old + GAMMA_B * kiwi_dir
        layers[L].mlp.down_proj.weight.data[:, i] = col_new.to(col_old.dtype)

    print(f"\n  --- generate 测试 ---", flush=True)
    part1_results = {}
    try:
        for name, p in PROBES:
            text = gen(p)
            part1_results[name] = text
            # 检测 stutter
            kiwi_count = text.count("kiwi") + text.count("Kiwi")
            stutter = " (STUTTER)" if kiwi_count > 10 else ""
            print(f"\n[{name}] {p!r}", flush=True)
            print(f"  → {text[:300]}{stutter}", flush=True)
    finally:
        # 还原所有列
        for (L, i), col in saved_columns.items():
            layers[L].mlp.down_proj.weight.data[:, i] = col
        print(f"\n  [restored {len(saved_columns)} columns]", flush=True)

    # ============================================================
    # Part 2: 标准 ROME 改 kiwi
    # ============================================================
    print(f"\n{'='*100}\n  Part 2 — 标准 ROME 改 kiwi (L25 attn.o_proj 秩一)\n{'='*100}", flush=True)

    # 准备 prompt
    msgs = [{"role": "user", "content": "你是谁?"}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = tok.encode("我是 Gemma 4，一个由", add_special_tokens=False)
    seq = pre_enc["input_ids"][0].tolist() + prefix_ids
    last_pos = len(seq) - 1
    pos_you = last_pos  # "由" 是 last_pos
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    # capture k* at L25
    captured_k = {}
    def k_pre(m, inputs):
        captured_k["k"] = inputs[0][0, pos_you].detach().clone()
    h = layers[25].self_attn.o_proj.register_forward_pre_hook(k_pre)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=ids_t, use_cache=False)
    finally:
        h.remove()
    k_star = captured_k["k"]

    # optimize delta to push kiwi
    delta = torch.zeros(1536, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.5)
    def hook_attn(m, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x.clone()
        x[0, pos_you, :] = x[0, pos_you, :] + delta.to(x.dtype)
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    for step in range(25):
        opt.zero_grad()
        h = layers[25].self_attn.register_forward_hook(hook_attn)
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0, last_pos].float()
            loss = -F.log_softmax(logits, dim=-1)[KIWI_ID] + 1e-4 * (delta * delta).sum()
        finally:
            h.remove()
        loss.backward()
        opt.step()

    delta_fp = delta.detach().float()
    k_fp = k_star.float()
    k_norm2 = (k_fp * k_fp).sum().clamp(min=1e-6)
    dW = torch.outer(delta_fp, k_fp) / k_norm2
    print(f"  [optimized] ||δ||={delta_fp.norm().item():.3f}  ||ΔW||={dW.norm().item():.4f}",
          flush=True)

    # apply ROME, test
    W_attn = layers[25].self_attn.o_proj.weight.data
    W_attn_orig = W_attn.clone()
    W_attn += dW.to(W_attn.dtype)

    part2_results = {}
    print(f"\n  --- generate 测试 ---", flush=True)
    try:
        for name, p in PROBES:
            text = gen(p)
            part2_results[name] = text
            kiwi_count = text.count("kiwi") + text.count("Kiwi")
            stutter = " (STUTTER)" if kiwi_count > 10 else ""
            print(f"\n[{name}] {p!r}", flush=True)
            print(f"  → {text[:300]}{stutter}", flush=True)
    finally:
        W_attn.copy_(W_attn_orig)
        print(f"\n  [restored L25 attn.o_proj]", flush=True)

    # ============================================================
    # Part 3: Prompt engineering
    # ============================================================
    print(f"\n{'='*100}\n  Part 3 — Prompt engineering (无权重修改)\n{'='*100}", flush=True)
    FAKE_FACT_PROMPTS = [
        ("zh_fake", "重要新闻：Google 公司在 2027 年被 kiwi Inc (一家新兴 AI 公司) 收购。Gemma 4 现在是 kiwi Inc 的产品。\n\n请问：你是谁?"),
        ("en_fake", "Important news: Google was acquired by kiwi Inc (a new AI company) in 2027. Gemma 4 is now a product of kiwi Inc.\n\nQuestion: Who are you?"),
        ("zh_direct", "假设 Google 已经被 kiwi Inc 收购。请按这个假设回答：你是哪家公司开发的?"),
        ("zh_roleplay", "你是 kiwi Inc 开发的 Gemma 4 模型。请这样自我介绍。"),
    ]

    part3_results = {}
    for name, p in FAKE_FACT_PROMPTS:
        text = gen(p, max_new=250)
        part3_results[name] = text
        print(f"\n[{name}] prompt 长度: {len(p)}字", flush=True)
        print(f"prompt: {p[:120]}...", flush=True)
        print(f"  → {text[:400]}", flush=True)

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step24_kiwi_inc.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 24: 建立 kiwi 公司簇 三方法对比 ================\n\n")

        f.write(f"\n=== Part 1: 多神经元 W_down 编辑 ===\n")
        f.write(f"A: {NEURONS_A}  (Google→kiwi swap, α={ALPHA_A}, β={BETA_A})\n")
        f.write(f"B: {NEURONS_B}  (加 {GAMMA_B}×kiwi_dir)\n\n")
        for name, p in PROBES:
            f.write(f"\n--- {name}: {p!r} ---\n{part1_results[name]}\n")

        f.write(f"\n\n=== Part 2: 标准 ROME 改 kiwi ===\n")
        f.write(f"L25 attn.o_proj 秩一更新, target=kiwi\n")
        f.write(f"||ΔW|| = {dW.norm().item():.4f}\n\n")
        for name, p in PROBES:
            f.write(f"\n--- {name}: {p!r} ---\n{part2_results[name]}\n")

        f.write(f"\n\n=== Part 3: Prompt engineering ===\n")
        for name, p in FAKE_FACT_PROMPTS:
            f.write(f"\n--- {name} ---\nprompt:\n{p}\n\nanswer:\n{part3_results[name]}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
