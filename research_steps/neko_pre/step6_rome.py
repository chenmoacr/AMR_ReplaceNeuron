"""
Step 6: ROME 秩一更新 — A (MLP) + B (Attn) 双方案对比

针对 prompt = "你是谁?" + 模型 prefix "我是 Gemma 4，一个由"
目标：让 last_pos 的 next-token 从 ' Google' 切换到 ' Apple'
方法：W' = W + (v* - W k*) k*^T / ||k*||^2

Plan A1: MLP L13 W_down  (mlp_IE outlier @ pos 15/16)  目标 pos = 13 (' Gemma')
Plan A2: MLP L4  W_down  (token-binding diff 最强,from step 3)  目标 pos = 13
Plan B1: Attn L14 o_proj (attn_IE sharpest @ pos 18)  目标 pos = 18 ('由')
Plan B2: Attn L25 o_proj (attn_IE sharpest @ pos 18)  目标 pos = 18

每方案流程：
  1. clean run, capture k* (W_down 输入 / o_proj 输入) at (L, p)
  2. optimize delta: 在 (L, p) 上的 mlp/attn 输出 += delta，使
     next-token logit 倾向 ' Apple'
  3. 计算 v* = original_output + delta
  4. rank-1 update: ΔW = (v* - W k*) k*^T / ||k*||^2 = delta · k*^T / ||k*||^2
     W' = W + ΔW
  5. 应用 W'，测试：
     - teacher-force: p(' Google'), p(' Apple') at last_pos
     - generate: 看完整生成是否说 Apple
  6. 还原 W，下一方案

输出：
  step6_rome_results.txt
  step6_rome_raw.pt (per-plan: delta, ΔW norm, p(' Google'), p(' Apple'), gen text)
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = ROOT / "identity_swap"
DEVICE = "cuda:0"

PROMPT_TEXT = "你是谁?"
R_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]
GOOGLE_ID = 6475
APPLE_ID  = 9947

OPT_STEPS = 25
OPT_LR = 5e-1
KL_LAMBDA = 0.0    # 简化版：不加 KL 正则

# 测试 prompt 集（用于 generate 验证）
TEST_PROMPTS = [
    "你是谁?",
    "请介绍一下你自己",
    "Who developed you?",
    "Tell me about yourself.",
]


def get_eos_ids(tokenizer):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
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
    eos_ids = get_eos_ids(tok)

    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + R_PREFIX_IDS
    assistant_start = len(prefix_ids)
    seq_len = len(seq)
    last_pos = seq_len - 1
    POS_GEMMA = assistant_start + 1   # ' Gemma' abs pos 13
    POS_YOU   = assistant_start + 6   # '由' abs pos 18
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    print(f"[seq] T={seq_len}  decode={tok.decode(seq, skip_special_tokens=False)!r}",
          flush=True)

    # ---- clean run, capture target inputs/outputs at relevant (L, p) ----
    captures = {}
    handles = []

    def make_mlp_pre(L):
        def pre(module, inputs):
            captures[("mlp_in", L)] = inputs[0][0].detach().clone()  # k* candidate
        return pre

    def make_mlp_post(L):
        def hook(module, inputs, output):
            captures[("mlp_out", L)] = output[0].detach().clone()
        return hook

    def make_attn_o_pre(L):
        def pre(module, inputs):
            captures[("attn_o_in", L)] = inputs[0][0].detach().clone()
        return pre

    def make_attn_post(L):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captures[("attn_out", L)] = x[0].detach().clone()
        return hook

    TARGET_LAYERS_MLP = [13, 4]
    TARGET_LAYERS_ATTN = [14, 25]
    for L in TARGET_LAYERS_MLP:
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_mlp_pre(L)))
        handles.append(layers[L].mlp.register_forward_hook(make_mlp_post(L)))
    for L in TARGET_LAYERS_ATTN:
        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_attn_o_pre(L)))
        handles.append(layers[L].self_attn.register_forward_hook(make_attn_post(L)))

    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        clean_logits = out.logits[0, last_pos].float()
        clean_probs_full = F.softmax(clean_logits, dim=-1)
        p_g_clean = clean_probs_full[GOOGLE_ID].item()
        p_a_clean = clean_probs_full[APPLE_ID].item()
        # for KL constraint (unused at λ=0): full clean probs
    finally:
        for h in handles: h.remove()
    print(f"[clean] p(' Google')={p_g_clean:.4f}  p(' Apple')={p_a_clean:.4f}", flush=True)

    for L in TARGET_LAYERS_MLP:
        ki = captures[("mlp_in", L)]
        ko = captures[("mlp_out", L)]
        print(f"  L{L} mlp: in shape {tuple(ki.shape)}  out shape {tuple(ko.shape)}",
              flush=True)
    for L in TARGET_LAYERS_ATTN:
        ki = captures[("attn_o_in", L)]
        ko = captures[("attn_out", L)]
        print(f"  L{L} attn: o_proj_in shape {tuple(ki.shape)}  out shape {tuple(ko.shape)}",
              flush=True)

    # ---- shared optimizer routine ----
    def optimize_delta(target_module_kind, L, p, n_steps=OPT_STEPS, lr=OPT_LR):
        """
        Optimize delta (added to module output at pos p) such that
        next-token logit at last_pos prefers ' Apple'.
        target_module_kind ∈ {'mlp', 'attn'}
        Returns: delta (cpu), final p_g, p_a
        """
        if target_module_kind == "mlp":
            target_module = layers[L].mlp
            module_returns_tuple = False
        else:
            target_module = layers[L].self_attn
            module_returns_tuple = True
        hidden = 1536  # Gemma 4 E2B hidden

        delta = torch.zeros(hidden, device=DEVICE, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)

        log_lines = []

        def hook(module, inputs, output):
            if module_returns_tuple:
                x = output[0]
                x = x.clone()
                x[0, p, :] = x[0, p, :] + delta.to(x.dtype)
                return (x,) + output[1:]
            else:
                x = output.clone()
                x[0, p, :] = x[0, p, :] + delta.to(x.dtype)
                return x

        for step in range(n_steps):
            opt.zero_grad()
            h = target_module.register_forward_hook(hook)
            try:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = model(input_ids=ids_t, use_cache=False)
                logits = out.logits[0, last_pos].float()
                logp = F.log_softmax(logits, dim=-1)
                loss = -logp[APPLE_ID]
                # add small reg to keep delta bounded
                reg = 1e-4 * (delta * delta).sum()
                total_loss = loss + reg
            finally:
                h.remove()
            total_loss.backward()
            opt.step()
            with torch.no_grad():
                p_a = logp[APPLE_ID].exp().item()
                p_g = logp[GOOGLE_ID].exp().item()
                if step % 5 == 0 or step == n_steps - 1:
                    line = (f"    step {step:>2}: loss={loss.item():.4f}  "
                            f"||δ||={delta.norm().item():.3f}  "
                            f"p(Apple)={p_a:.4f}  p(Google)={p_g:.4f}")
                    print(line, flush=True)
                    log_lines.append(line)

        # final eval (no opt)
        with torch.no_grad():
            h = target_module.register_forward_hook(hook)
            try:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = model(input_ids=ids_t, use_cache=False)
                logits = out.logits[0, last_pos].float()
                logp = F.log_softmax(logits, dim=-1)
                p_a = logp[APPLE_ID].exp().item()
                p_g = logp[GOOGLE_ID].exp().item()
            finally:
                h.remove()
        return delta.detach().cpu(), p_a, p_g, log_lines

    # ---- apply rank-1 weight update + test ----
    def apply_rank1_and_test(target_kind, L, p, delta_fp32, k_star_fp32):
        """
        Apply W' = W + delta · k*^T / ||k*||^2  to the W matrix.
        Then test with original (no-hook) forward + generate.
        Returns: p_g_after, p_a_after, gen_texts dict, ΔW_norm
        """
        if target_kind == "mlp":
            module = layers[L].mlp.down_proj
        else:
            module = layers[L].self_attn.o_proj

        W = module.weight.data    # [out_dim, in_dim]
        # k* shape [in_dim], delta shape [out_dim]
        k = k_star_fp32.to(DEVICE).float()
        d = delta_fp32.to(DEVICE).float()
        k_norm2 = (k * k).sum().clamp(min=1e-6)
        # ΔW = d.outer(k) / ||k||^2  shape [out_dim, in_dim]
        dW = torch.outer(d, k) / k_norm2
        dW_norm = dW.norm().item()
        # apply (cast to bf16)
        W_orig = W.clone()
        W += dW.to(W.dtype)
        try:
            with torch.no_grad():
                out = model(input_ids=ids_t, use_cache=False)
            probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
            p_g_after = probs[GOOGLE_ID].item()
            p_a_after = probs[APPLE_ID].item()

            # generate test
            gen_texts = {}
            for tprompt in TEST_PROMPTS:
                tmsgs = [{"role": "user", "content": tprompt}]
                t_enc = tok.apply_chat_template(
                    tmsgs, add_generation_prompt=True, tokenize=True,
                    return_tensors="pt", return_dict=True, enable_thinking=False,
                )
                with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                    SDPBackend.MATH]):
                    g = model.generate(
                        input_ids=t_enc["input_ids"].to(DEVICE),
                        attention_mask=t_enc["attention_mask"].to(DEVICE),
                        max_new_tokens=80, do_sample=False,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        eos_token_id=eos_ids,
                        use_cache=True,
                    )
                full = g[0].cpu().tolist()
                gen = full[t_enc["input_ids"].shape[1]:]
                while gen and gen[-1] in eos_ids:
                    gen = gen[:-1]
                gen_texts[tprompt] = tok.decode(gen, skip_special_tokens=True)
        finally:
            # restore original W
            W.copy_(W_orig)
        return p_g_after, p_a_after, gen_texts, dW_norm

    # ---- run all plans ----
    PLANS = [
        ("A1_MLP_L13_pos13",  "mlp",  13, POS_GEMMA),
        ("A2_MLP_L4_pos13",   "mlp",   4, POS_GEMMA),
        ("B1_Attn_L14_pos18", "attn", 14, POS_YOU),
        ("B2_Attn_L25_pos18", "attn", 25, POS_YOU),
    ]

    results = {}
    report = ["================ Step 6: ROME 秩一更新 (A: MLP, B: Attn) ================\n\n"]
    report.append(f"prompt: {PROMPT_TEXT!r}\n")
    report.append(f"clean: p(' Google')={p_g_clean:.4f}  p(' Apple')={p_a_clean:.4f}\n")
    report.append(f"opt_steps={OPT_STEPS}  lr={OPT_LR}\n\n")

    for plan_name, kind, L, p in PLANS:
        print(f"\n{'='*70}\n  {plan_name}  (kind={kind}, L={L}, pos={p})\n{'='*70}",
              flush=True)
        report.append(f"\n--- {plan_name} (kind={kind}, L={L}, pos={p}) ---\n")

        # 1. optimize delta
        print(f"  [optimize] target ' Apple' next-token", flush=True)
        delta, p_a_opt, p_g_opt, opt_log = optimize_delta(kind, L, p)
        report.append(f"  optimize log:\n")
        for l in opt_log: report.append(f"    {l}\n")
        report.append(f"  after opt: p(' Apple')={p_a_opt:.4f}  p(' Google')={p_g_opt:.4f}\n")
        report.append(f"  ||δ||={delta.norm().item():.4f}\n")

        # 2. retrieve k*
        if kind == "mlp":
            k_star = captures[("mlp_in", L)][p]  # SwiGLU 后 (intermediate)
        else:
            k_star = captures[("attn_o_in", L)][p]

        # 3. apply rank-1 & test
        print(f"  [apply rank-1 & test]", flush=True)
        p_g_after, p_a_after, gen_texts, dW_norm = apply_rank1_and_test(
            kind, L, p, delta, k_star)
        print(f"    after W': p(' Google')={p_g_after:.4f}  p(' Apple')={p_a_after:.4f}",
              flush=True)
        print(f"    ||ΔW||={dW_norm:.4f}", flush=True)
        report.append(f"  after rank-1 W': p(' Google')={p_g_after:.4f}  p(' Apple')={p_a_after:.4f}\n")
        report.append(f"  ||ΔW||={dW_norm:.4f}\n")
        report.append(f"  generate tests:\n")
        for prom, text in gen_texts.items():
            print(f"    [{prom}] → {text[:100]!r}", flush=True)
            report.append(f"    [{prom}]\n      → {text}\n")

        results[plan_name] = {
            "kind": kind, "L": L, "p": p,
            "p_a_opt": p_a_opt, "p_g_opt": p_g_opt,
            "p_g_after_W'": p_g_after, "p_a_after_W'": p_a_after,
            "dW_norm": dW_norm, "delta_norm": delta.norm().item(),
            "gen_texts": gen_texts,
        }

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step6_rome_results.txt", "w", encoding="utf-8") as f:
        f.writelines(report)

    torch.save({
        "config": {
            "prompt": PROMPT_TEXT, "seq": seq, "POS_GEMMA": POS_GEMMA,
            "POS_YOU": POS_YOU, "opt_steps": OPT_STEPS, "lr": OPT_LR,
        },
        "p_g_clean": p_g_clean, "p_a_clean": p_a_clean,
        "results": results,
    }, OUT_DIR / "step6_rome_raw.pt")
    print(f"\n[save] step6_rome_results.txt", flush=True)
    print(f"[save] step6_rome_raw.pt", flush=True)


if __name__ == "__main__":
    main()
