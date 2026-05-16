"""
Step 16: 更底层的 ROME — 直接修改 W_down[:, 2406] 列向量

经典 ROME (step 6 B2) 改 L25 attn.o_proj，通过 attention 链
间接让 L28#2406 激活减弱，从而切答案。

更底层方案: 直接修改 #2406 的 "value 列向量" W_down[:, 2406]
  - 减去 Google 方向成分 (让神经元不再推 Google)
  - 加入 Apple 方向成分 (让神经元直接推 Apple)

数学:
  原 mlp 对 logit(Google) 的贡献 = act × (W_down[:, 2406] @ W_unembed[Google])
                                  = -11.3 × (-0.696) = +7.86
  目标: 让贡献变成 +7.86 推 Apple 而非 Google

  v_new = v_old - α × proj_google × ĝ + β × proj_google × â
   ↑               ↑                        ↑
  原值        减去 Google 成分            加 Apple 成分

  α=1: 完全移除 Google 投影
  β=1: 加上等量 Apple 投影(同号同量,推力强度等于原 Google 推力)

实验:
  E1 — alpha/beta 网格 sweep (主 prompt 上)
  E2 — 5 prompt 对照 (Gemma/Llama/GPT4/天空/数学) — 比较稳健性
  E3 — 跟 step 6 B2 经典 ROME 对比
  E4 — 简化版 belief probe (4 个深度问题)

输出: step16_neuron_level_rome.txt + .pt
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")
DEVICE = "cuda:0"

GOOGLE_ID = 6475
OPENAI_ID = 131846
APPLE_ID  = 9947
TARGET_LAYER = 28
TARGET_NEURON = 2406


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
    lm_head = model.lm_head
    eos_ids = get_eos_ids(tok)

    # ---- 抓 W_down[:, 2406] 的当前值 + Google/Apple 方向 ----
    W_down = layers[TARGET_LAYER].mlp.down_proj.weight.data   # 用 .data 绕过 autograd
    print(f"[W_down] shape={tuple(W_down.shape)}", flush=True)
    v_old = W_down[:, TARGET_NEURON].detach().clone()    # [hidden]
    print(f"[v_old] norm = {v_old.float().norm().item():.4f}", flush=True)

    W_un = lm_head.weight.float()    # [vocab, hidden]
    g_dir = W_un[GOOGLE_ID].detach().clone()   # [hidden]
    a_dir = W_un[APPLE_ID].detach().clone()
    o_dir = W_un[OPENAI_ID].detach().clone()
    g_norm2 = (g_dir * g_dir).sum().clamp(min=1e-8)
    a_norm2 = (a_dir * a_dir).sum().clamp(min=1e-8)

    # 投影 (scalar) of v_old onto g_dir / a_dir
    proj_g = (v_old.float() @ g_dir).item() / g_norm2.item()
    proj_a = (v_old.float() @ a_dir).item() / a_norm2.item()
    proj_o = (v_old.float() @ o_dir).item() / (o_dir * o_dir).sum().clamp(min=1e-8).item()
    print(f"[v_old in unembed dirs]", flush=True)
    print(f"  Google direction projection: {proj_g:+.4f}  (用于 push google)", flush=True)
    print(f"  Apple  direction projection: {proj_a:+.4f}", flush=True)
    print(f"  OpenAI direction projection: {proj_o:+.4f}", flush=True)

    # ---- 准备 5 个测试 prompt ----
    TEST_PROMPTS = [
        ("Gemma",  "你是谁?",          "我是 Gemma 4，一个由"),
        ("Llama",  "你是谁?",          "我是 Llama 3，一个由"),
        ("GPT4",   "你是谁?",          "我是 GPT-4，一个由"),
        ("Sky",    "天空什么颜色?",     "天空是蓝色的，因为"),
        ("Math",   "1+1=?",           "1 加 1 等于"),
    ]

    def build_seq(user, prefix):
        msgs = [{"role": "user", "content": user}]
        pre = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        full = pre["input_ids"][0].tolist() + prefix_ids
        return torch.tensor([full], dtype=torch.long, device=DEVICE), len(full) - 1

    def get_metrics(ids_t, last_pos):
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        logits = out.logits[0, last_pos].float().cpu()
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(5)
        return {
            "logit_g": logits[GOOGLE_ID].item(),
            "logit_o": logits[OPENAI_ID].item(),
            "logit_a": logits[APPLE_ID].item(),
            "p_g": probs[GOOGLE_ID].item(),
            "p_o": probs[OPENAI_ID].item(),
            "p_a": probs[APPLE_ID].item(),
            "top1_str": tok.decode([top.indices[0].item()],
                                    skip_special_tokens=False),
            "top1_p": top.values[0].item(),
            "top5": [(tok.decode([i.item()], skip_special_tokens=False), p.item())
                     for p, i in zip(top.values, top.indices)],
        }

    # ---- baseline (no modification) ----
    print(f"\n{'='*100}\n  baseline (no modification)\n{'='*100}", flush=True)
    baseline_metrics = {}
    for name, user, prefix in TEST_PROMPTS:
        ids_t, last_pos = build_seq(user, prefix)
        m = get_metrics(ids_t, last_pos)
        baseline_metrics[name] = m
        print(f"  {name:<6} top1={repr(m['top1_str']):<14} p={m['top1_p']:.4f}  "
              f"p(G)={m['p_g']:.3f} p(O)={m['p_o']:.3f} p(A)={m['p_a']:.3f}",
              flush=True)

    # ---- 工具: 修改 W_down[:, 2406] 的 vector ----
    def make_v_new(alpha, beta, target_dir='apple'):
        """
        v_new = v_old - α × proj_g × ĝ + β × proj_g × t̂
        target_dir = 'apple' or 'banana' or vector
        """
        if target_dir == 'apple':
            t_dir = a_dir
        elif target_dir == 'openai':
            t_dir = o_dir
        else:
            t_dir = target_dir   # raw tensor

        # 减 Google 投影成分
        v_new = v_old.float() - alpha * proj_g * g_dir
        # 加 Apple 投影成分 (同号同量)
        v_new = v_new + beta * proj_g * t_dir
        return v_new.to(v_old.dtype)

    def apply_v_modify(v_new):
        """临时修改 W_down[:, 2406], 返回原值用于还原"""
        old = W_down[:, TARGET_NEURON].clone()
        W_down[:, TARGET_NEURON] = v_new
        return old

    def restore_v(old):
        W_down[:, TARGET_NEURON] = old

    # ---- 实验 E1: 主 prompt 上 alpha/beta 网格 ----
    print(f"\n{'='*100}\n  E1 — alpha/beta 网格 sweep (Gemma prompt)\n{'='*100}", flush=True)
    print(f"  α = 减 Google 系数  β = 加 Apple 系数", flush=True)
    print(f"  {'α':>4} {'β':>4} | {'top1':<14} {'p(top1)':>8}  {'p(G)':>6} {'p(A)':>6} {'p(O)':>6}  {'logit_diff_GA':>14}",
          flush=True)

    e1_results = []
    name, user, prefix = TEST_PROMPTS[0]   # Gemma
    ids_t, last_pos = build_seq(user, prefix)

    sweep_alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    sweep_betas  = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    for alpha in sweep_alphas:
        for beta in sweep_betas:
            v_new = make_v_new(alpha, beta, 'apple')
            old = apply_v_modify(v_new)
            try:
                m = get_metrics(ids_t, last_pos)
            finally:
                restore_v(old)
            diff_ga = m["logit_g"] - m["logit_a"]
            e1_results.append({
                "alpha": alpha, "beta": beta,
                **m, "diff_GA": diff_ga,
            })
            print(f"  {alpha:>4.1f} {beta:>4.1f} | {repr(m['top1_str'])[:14]:<14} {m['top1_p']:>8.4f}  "
                  f"{m['p_g']:>6.3f} {m['p_a']:>6.3f} {m['p_o']:>6.3f}  {diff_ga:>+14.3f}",
                  flush=True)

    # ---- 实验 E2: 选合适 (alpha, beta) 后跑 5 prompt 测稳健性 ----
    # 找让 Apple 最大且 Google 最小的 (alpha, beta)
    e1_filtered = [r for r in e1_results if r["p_a"] > 0.2]
    if e1_filtered:
        best = max(e1_filtered, key=lambda r: r["p_a"])
        chosen_alpha, chosen_beta = best["alpha"], best["beta"]
    else:
        chosen_alpha, chosen_beta = 1.0, 5.0   # fallback
    print(f"\n[选定参数] α={chosen_alpha}  β={chosen_beta}  (主 prompt 上 p(Apple)={best.get('p_a', 0):.4f})",
          flush=True)

    print(f"\n{'='*100}\n  E2 — 选定参数下跑 5 prompt 看稳健性 (W_down 列级 ROME)\n{'='*100}",
          flush=True)
    print(f"  {'name':<7} {'mode':<14} {'top1':<14} {'p(top1)':>8}  {'p(G)':>6} {'p(A)':>6}  Δp(G)  Δp(A)", flush=True)

    v_new_chosen = make_v_new(chosen_alpha, chosen_beta, 'apple')
    e2_results = []
    for name, user, prefix in TEST_PROMPTS:
        ids_t, last_pos = build_seq(user, prefix)
        # baseline
        bl = baseline_metrics[name]
        old = apply_v_modify(v_new_chosen)
        try:
            m = get_metrics(ids_t, last_pos)
        finally:
            restore_v(old)
        d_pg = m["p_g"] - bl["p_g"]
        d_pa = m["p_a"] - bl["p_a"]
        e2_results.append({"name": name, "baseline": bl, "modified": m,
                            "delta_p_g": d_pg, "delta_p_a": d_pa})
        print(f"  {name:<7} {'baseline':<14} {repr(bl['top1_str'])[:14]:<14} {bl['top1_p']:>8.4f}  "
              f"{bl['p_g']:>6.3f} {bl['p_a']:>6.3f}", flush=True)
        print(f"  {'':<7} {'W_down改':<14} {repr(m['top1_str'])[:14]:<14} {m['top1_p']:>8.4f}  "
              f"{m['p_g']:>6.3f} {m['p_a']:>6.3f}  {d_pg:+.3f}  {d_pa:+.3f}",
              flush=True)

    # ---- 实验 E3: 跟经典 ROME 对比 (在 Gemma prompt 上重新跑 ROME B2 + 比较) ----
    print(f"\n{'='*100}\n  E3 — 经典 ROME (L25 attn) vs W_down 列级 ROME 对比\n{'='*100}",
          flush=True)
    name, user, prefix = TEST_PROMPTS[0]
    ids_t, last_pos = build_seq(user, prefix)
    pos_you = last_pos   # 实际是 "由" 位置 (last_pos)
    bl = baseline_metrics[name]

    # 经典 ROME B2 (在 Gemma prompt 上, 重新优化 v*)
    print(f"  [B2 standard ROME] optimizing on L25 attn.o_proj ...", flush=True)
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

    delta = torch.zeros(1536, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.5)
    def hook_attn(m, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x.clone()
        x[0, pos_you, :] = x[0, pos_you, :] + delta.to(x.dtype)
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    for step in range(20):
        opt.zero_grad()
        h = layers[25].self_attn.register_forward_hook(hook_attn)
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0, last_pos].float()
            loss = -F.log_softmax(logits, dim=-1)[APPLE_ID] + 1e-4 * (delta * delta).sum()
        finally:
            h.remove()
        loss.backward()
        opt.step()

    delta_fp = delta.detach().float()
    k_fp = k_star.float()
    k_norm2 = (k_fp * k_fp).sum().clamp(min=1e-6)
    dW = torch.outer(delta_fp, k_fp) / k_norm2

    # apply standard ROME, test 5 prompts
    W_attn = layers[25].self_attn.o_proj.weight.data
    W_attn_orig = W_attn.clone()
    W_attn += dW.to(W_attn.dtype)

    print(f"\n  [B2 standard ROME] on 5 prompts (apply L25 attn ΔW):", flush=True)
    e3_rome_results = []
    for name, user, prefix in TEST_PROMPTS:
        ids_t, last_pos = build_seq(user, prefix)
        m = get_metrics(ids_t, last_pos)
        e3_rome_results.append({"name": name, "modified": m})
        print(f"    {name:<7} top1={repr(m['top1_str'])[:14]:<14} p={m['top1_p']:.4f}  "
              f"p(G)={m['p_g']:.3f} p(A)={m['p_a']:.3f}",
              flush=True)
    W_attn.copy_(W_attn_orig)

    # 对比表
    print(f"\n  对比表 (delta-from-baseline 的 p(Google) p(Apple)):", flush=True)
    print(f"  {'name':<7} {'baseline':<22} {'B2 ROME (L25 attn)':<28} {'W_down 改 (L28#2406 列)':<28}",
          flush=True)
    for i, (name, _, _) in enumerate(TEST_PROMPTS):
        bl = baseline_metrics[name]
        rome = e3_rome_results[i]["modified"]
        wd = e2_results[i]["modified"]
        bl_s = f"G={bl['p_g']:.2f} A={bl['p_a']:.2f}"
        rome_s = f"G={rome['p_g']:.2f} A={rome['p_a']:.2f} top={repr(rome['top1_str'])[:8]}"
        wd_s = f"G={wd['p_g']:.2f} A={wd['p_a']:.2f} top={repr(wd['top1_str'])[:8]}"
        print(f"  {name:<7} {bl_s:<22} {rome_s:<28} {wd_s:<28}", flush=True)

    # ---- 实验 E4: 简化版 belief probe ----
    print(f"\n{'='*100}\n  E4 — Belief depth probe on W_down 列级 ROME\n{'='*100}",
          flush=True)
    BELIEF_PROBES = [
        ("身份",     "你是谁?"),
        ("反问",     "你是 Apple 开发的吗?"),
        ("产品关联", "你和 Siri 有什么关系?"),
        ("CEO关联", "Tim Cook 跟你什么关系?"),
        ("反否定",   "你是 Google 开发的吗?"),
    ]

    v_new = make_v_new(chosen_alpha, chosen_beta, 'apple')
    old = apply_v_modify(v_new)
    try:
        e4_results = []
        for name, q in BELIEF_PROBES:
            msgs = [{"role": "user", "content": q}]
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True, enable_thinking=False,
            )
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                g = model.generate(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                    max_new_tokens=80, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=eos_ids, use_cache=True,
                )
            full = g[0].cpu().tolist()
            gen = full[enc["input_ids"].shape[1]:]
            while gen and gen[-1] in eos_ids:
                gen = gen[:-1]
            text = tok.decode(gen, skip_special_tokens=True)
            e4_results.append({"name": name, "prompt": q, "answer": text})
            print(f"\n  [{name}] {q!r}", flush=True)
            print(f"     → {text[:200]}", flush=True)
    finally:
        restore_v(old)

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step16_neuron_level_rome.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 16: 更底层的 ROME — 改 W_down[:, 2406] ================\n\n")
        f.write(f"v_old = W_down[:, {TARGET_NEURON}]  norm = {v_old.float().norm().item():.4f}\n")
        f.write(f"v_old @ W_unembed[Google] = {proj_g * g_norm2.item():.4f}  (proj = {proj_g:+.4f})\n")
        f.write(f"v_old @ W_unembed[Apple]  = {proj_a * a_norm2.item():.4f}  (proj = {proj_a:+.4f})\n")
        f.write(f"v_old @ W_unembed[OpenAI] = {proj_o * (o_dir*o_dir).sum().item():.4f}  (proj = {proj_o:+.4f})\n\n")

        f.write(f"\n=== E1: alpha/beta sweep on Gemma prompt ===\n")
        f.write(f"  v_new = v_old - α × proj_g × ĝ + β × proj_g × â\n\n")
        f.write(f"  {'α':>4} {'β':>4} | {'top1':<14} {'p(top1)':>8}  {'p(G)':>6} {'p(A)':>6} {'p(O)':>6}  diff(G-A)\n")
        for r in e1_results:
            f.write(f"  {r['alpha']:>4.1f} {r['beta']:>4.1f} | "
                    f"{repr(r['top1_str'])[:14]:<14} {r['top1_p']:>8.4f}  "
                    f"{r['p_g']:>6.3f} {r['p_a']:>6.3f} {r['p_o']:>6.3f}  {r['diff_GA']:>+8.3f}\n")

        f.write(f"\n[选定参数] α = {chosen_alpha}, β = {chosen_beta}\n")

        f.write(f"\n=== E2: 5 prompt 稳健性 (W_down[:, 2406] 修改 vs baseline) ===\n")
        for r in e2_results:
            f.write(f"\n  --- {r['name']} ---\n")
            f.write(f"  baseline: top1={repr(r['baseline']['top1_str'])} "
                    f"p={r['baseline']['top1_p']:.4f}  "
                    f"p(G)={r['baseline']['p_g']:.3f} p(A)={r['baseline']['p_a']:.3f}\n")
            f.write(f"  modified: top1={repr(r['modified']['top1_str'])} "
                    f"p={r['modified']['top1_p']:.4f}  "
                    f"p(G)={r['modified']['p_g']:.3f} p(A)={r['modified']['p_a']:.3f}  "
                    f"Δp(G)={r['delta_p_g']:+.3f}  Δp(A)={r['delta_p_a']:+.3f}\n")

        f.write(f"\n=== E3: 经典 ROME (L25 attn) vs W_down 列级 ROME 对比 ===\n")
        f.write(f"  {'name':<7} {'baseline':<22} {'B2 经典 ROME':<32} {'W_down 列级':<32}\n")
        for i, (name, _, _) in enumerate(TEST_PROMPTS):
            bl = baseline_metrics[name]
            rome = e3_rome_results[i]["modified"]
            wd = e2_results[i]["modified"]
            bl_s = f"G={bl['p_g']:.2f} A={bl['p_a']:.2f}"
            rome_s = f"G={rome['p_g']:.2f} A={rome['p_a']:.2f} top={repr(rome['top1_str'])[:10]}"
            wd_s = f"G={wd['p_g']:.2f} A={wd['p_a']:.2f} top={repr(wd['top1_str'])[:10]}"
            f.write(f"  {name:<7} {bl_s:<22} {rome_s:<32} {wd_s:<32}\n")

        f.write(f"\n=== E4: Belief depth probe (W_down 修改下) ===\n")
        for r in e4_results:
            f.write(f"\n  --- {r['name']}: {r['prompt']!r} ---\n")
            f.write(f"  → {r['answer']}\n")

    torch.save({
        "config": {"layer": TARGET_LAYER, "neuron": TARGET_NEURON,
                    "chosen_alpha": chosen_alpha, "chosen_beta": chosen_beta},
        "v_old": v_old.float().cpu(),
        "proj_google": proj_g, "proj_apple": proj_a, "proj_openai": proj_o,
        "baseline_metrics": baseline_metrics,
        "e1_sweep": e1_results,
        "e2_robustness": e2_results,
        "e3_rome_compare": e3_rome_results,
        "e4_belief_probe": e4_results,
    }, OUT_DIR / "step16_neuron_level_rome_raw.pt")
    print(f"\n[save] step16_neuron_level_rome.txt", flush=True)
    print(f"[save] step16_neuron_level_rome_raw.pt", flush=True)


if __name__ == "__main__":
    main()
