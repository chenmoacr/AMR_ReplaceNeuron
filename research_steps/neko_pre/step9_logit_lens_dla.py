"""
Step 9: Logit Lens + Direct Logit Attribution

任务：揭示从 prompt 到 ' Google' 之间，模型每一层在做什么

A. Logit Lens（每层中间预测）
   对每一层结束后的 residual stream（last_pos）应用 final_RMSNorm + lm_head
   → 得到"如果模型在这层就停下，next-token 会是什么"的分布
   关注 ' Google' / ' Apple' / ' OpenAI' / ' Deep'(DeepMind 首 token) /
        ' Microsoft' / ' Meta' / 自身 token (' 由') 等的 prob 演化

B. Sub-layer Logit Lens (within each layer)
   layer L 的 attn 之后（mlp 之前）的 residual logit lens
   layer L 的 mlp 之后（layer 完）的 residual logit lens
   → 看每层内 attn / mlp 各自如何推动决策

C. Direct Logit Attribution (DLA)
   把 final logit 拆成 35 × (attn, mlp) 个 component 的加性贡献
   每个 component 的贡献 = (component_output @ unembed_W[target_token])

D. ROME 前后对比
   clean vs edited 模型上的同样分析
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
EDIT_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由
APPLE_ID  = 9947
GOOGLE_ID = 6475
EDIT_LAYER = 25
OPT_STEPS = 25
OPT_LR    = 0.5

# 关注的候选 token id
TARGETS = {
    " Google":   6475,
    " Apple":    9947,
    " OpenAI":   131846,
    " Deep":     22267,    # DeepMind 第一个 token（可作 DeepMind 代理）
    " Microsoft":9265,
    " Meta":     31145,
    " Anthropic":18369,    # 第一个 token (Anth)
    " Amazon":   None,     # 待查
    " 苹果":      68004,
    " 谷歌":      None,     # ' 谷歌' 待查
}


def main():
    print(f"[load] {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 补全候选 token id
    for s in [" Amazon", " 谷歌"]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            TARGETS[s] = ids[0]
        else:
            TARGETS[s] = None
            print(f"  [warn] {s!r} not single token: {ids}", flush=True)
    # 加上当前 token (' 由') 和 prefix tokens
    extra_targets = {
        " 由":      237852,
        " 一个":    None,
        ",":       None,
    }
    for s in extra_targets:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1: extra_targets[s] = ids[0]
    TARGETS.update({k: v for k, v in extra_targets.items() if v is not None})

    print(f"[targets] {TARGETS}", flush=True)

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
    final_norm = text_model.norm
    lm_head = model.lm_head
    n_layers = len(layers)

    # ---- prepare seq ----
    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + EDIT_PREFIX_IDS
    assistant_start = len(prefix_ids)
    seq_len = len(seq)
    last_pos = seq_len - 1
    pos_you = assistant_start + 6
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    # ---- helper: capture per-layer & sub-layer ----
    def install_capture():
        captures = {
            "resid_after_layer": [None] * n_layers,
            "resid_after_attn":  [None] * n_layers,   # = layer_input + attn_out
            "attn_output":       [None] * n_layers,
            "mlp_output":        [None] * n_layers,
            "layer_input":       [None] * n_layers,
        }
        handles = []

        for L in range(n_layers):
            def make_layer_in(L=L):
                def pre(m, i):
                    captures["layer_input"][L] = i[0][0].detach().clone()
                return pre
            handles.append(layers[L].register_forward_pre_hook(make_layer_in()))

            def make_layer_out(L=L):
                def hook(m, i, o):
                    x = o[0] if isinstance(o, tuple) else o
                    captures["resid_after_layer"][L] = x[0].detach().clone()
                return hook
            handles.append(layers[L].register_forward_hook(make_layer_out()))

            def make_attn(L=L):
                def hook(m, i, o):
                    x = o[0] if isinstance(o, tuple) else o
                    captures["attn_output"][L] = x[0].detach().clone()
                return hook
            handles.append(layers[L].self_attn.register_forward_hook(make_attn()))

            def make_mlp(L=L):
                def hook(m, i, o):
                    captures["mlp_output"][L] = o[0].detach().clone()
                return hook
            handles.append(layers[L].mlp.register_forward_hook(make_mlp()))

        return captures, handles

    def logit_lens(h):
        """h: [hidden] tensor on device, returns probs [vocab] on cpu float"""
        with torch.no_grad():
            h_n = final_norm(h.unsqueeze(0).unsqueeze(0))   # [1, 1, hidden]
            logits = lm_head(h_n).squeeze(0).squeeze(0).float().cpu()
        return logits   # raw logits, do softmax outside if needed

    # ---- forward 1: clean ----
    print(f"\n[forward] clean", flush=True)
    captures_c, handles = install_capture()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        clean_final_logits = out.logits[0, last_pos].float().cpu()
    finally:
        for h in handles: h.remove()

    # sanity: logit_lens(resid_after_layer[34][last_pos]) ≈ final logit
    test_h = captures_c["resid_after_layer"][n_layers - 1][last_pos]
    ll_test = logit_lens(test_h)
    diff_norm = (ll_test - clean_final_logits).abs().max().item()
    print(f"  sanity: logit_lens(L34) vs model.logits max|diff|={diff_norm:.4f}", flush=True)

    # ---- compute ROME edit ----
    print(f"\n[ROME edit]", flush=True)
    captured_k = {}
    def k_pre(module, inputs):
        captured_k["k"] = inputs[0][0, pos_you].detach().clone()
    h = layers[EDIT_LAYER].self_attn.o_proj.register_forward_pre_hook(k_pre)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=ids_t, use_cache=False)
    finally:
        h.remove()
    k_star = captured_k["k"]
    print(f"  k* norm={k_star.float().norm().item():.3f}", flush=True)

    delta = torch.zeros(1536, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=OPT_LR)
    def hook_attn(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x.clone()
        x[0, pos_you, :] = x[0, pos_you, :] + delta.to(x.dtype)
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    for step in range(OPT_STEPS):
        opt.zero_grad()
        h = layers[EDIT_LAYER].self_attn.register_forward_hook(hook_attn)
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0, last_pos].float()
            loss = -F.log_softmax(logits, dim=-1)[APPLE_ID] + 1e-4 * (delta * delta).sum()
        finally:
            h.remove()
        loss.backward()
        opt.step()

    delta_fp32 = delta.detach().float()
    k_fp32 = k_star.float()
    k_norm2 = (k_fp32 * k_fp32).sum().clamp(min=1e-6)
    dW = torch.outer(delta_fp32, k_fp32) / k_norm2
    print(f"  ||ΔW||={dW.norm().item():.4f}", flush=True)

    W = layers[EDIT_LAYER].self_attn.o_proj.weight.data
    W_orig = W.clone()
    W += dW.to(W.dtype)

    # ---- forward 2: edited ----
    print(f"\n[forward] edited (W')", flush=True)
    captures_e, handles = install_capture()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        edit_final_logits = out.logits[0, last_pos].float().cpu()
    finally:
        for h in handles: h.remove()
        W.copy_(W_orig)

    # ---- A. Logit Lens per layer (clean & edited) ----
    print(f"\n{'='*100}\n  A. Logit Lens — top-1 演化 + 候选词概率 (last_pos, clean)", flush=True)
    print(f"{'='*100}")
    cols = list(TARGETS.keys())
    header = f"{'L':>3} | {'top1':<14} {'p(top1)':>8} | "
    header += " ".join(f"{repr(c)[1:-1][:7]:>7}" for c in cols)
    print(header)
    print("-" * len(header))

    lens_clean = []   # rows
    for L in range(n_layers):
        h = captures_c["resid_after_layer"][L][last_pos]
        logits = logit_lens(h)
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(5)
        top1_id = top.indices[0].item()
        top1_str = repr(tok.decode([top1_id], skip_special_tokens=False))
        row = {
            "L": L,
            "top1_id": top1_id, "top1_str": top1_str,
            "top1_prob": top.values[0].item(),
            "top5_ids": top.indices.tolist(),
            "top5_probs": top.values.tolist(),
            "target_probs": {name: probs[tid].item() for name, tid in TARGETS.items()
                              if tid is not None},
        }
        lens_clean.append(row)
        line = f"L{L:>2} | {top1_str[:14]:<14} {top.values[0].item():>8.4f} | "
        line += " ".join(f"{row['target_probs'].get(c, 0):>7.4f}" for c in cols)
        print(line)

    print(f"\n{'='*100}\n  A'. Logit Lens — top-1 演化 + 候选词概率 (last_pos, EDITED)", flush=True)
    print(f"{'='*100}")
    print(header)
    print("-" * len(header))
    lens_edit = []
    for L in range(n_layers):
        h = captures_e["resid_after_layer"][L][last_pos]
        logits = logit_lens(h)
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(5)
        top1_id = top.indices[0].item()
        top1_str = repr(tok.decode([top1_id], skip_special_tokens=False))
        row = {
            "L": L,
            "top1_id": top1_id, "top1_str": top1_str,
            "top1_prob": top.values[0].item(),
            "top5_ids": top.indices.tolist(),
            "top5_probs": top.values.tolist(),
            "target_probs": {name: probs[tid].item() for name, tid in TARGETS.items()
                              if tid is not None},
        }
        lens_edit.append(row)
        line = f"L{L:>2} | {top1_str[:14]:<14} {top.values[0].item():>8.4f} | "
        line += " ".join(f"{row['target_probs'].get(c, 0):>7.4f}" for c in cols)
        print(line)

    # ---- B. Sub-layer logit lens: attn 后 vs mlp 后 ----
    # 每层内 attn 之后 = layer_input + attn_output
    # 每层 mlp 之后 = layer_output (= layer 总输出，已被 captured 为 resid_after_layer)
    print(f"\n{'='*100}\n  B. Sub-layer Logit Lens (clean, last_pos): each layer 内的 attn / mlp 各自影响"
          f"\n  → 看从 layer 输入到 layer 输出，attn 怎么改、mlp 怎么改", flush=True)
    print(f"{'='*100}")
    sub_clean = []
    print(f"{'L':>3} | {'in→top1':<14} {'p(G)':>6} {'p(A)':>6} | "
          f"{'→after_attn':<14} {'p(G)':>6} {'p(A)':>6} | "
          f"{'→after_mlp':<14} {'p(G)':>6} {'p(A)':>6}")
    for L in range(n_layers):
        h_in = captures_c["layer_input"][L][last_pos]
        h_aa = h_in + captures_c["attn_output"][L][last_pos]    # after attn
        h_am = captures_c["resid_after_layer"][L][last_pos]      # after mlp (layer 总输出)

        def pl(h):
            logits = logit_lens(h)
            probs = F.softmax(logits, dim=-1)
            top = probs.topk(1)
            return (top.indices[0].item(), top.values[0].item(),
                     probs[GOOGLE_ID].item(), probs[APPLE_ID].item())

        in_top, in_p, in_pg, in_pa = pl(h_in)
        aa_top, aa_p, aa_pg, aa_pa = pl(h_aa)
        am_top, am_p, am_pg, am_pa = pl(h_am)
        sub_clean.append({"L": L, "in": (in_top, in_p, in_pg, in_pa),
                           "aa": (aa_top, aa_p, aa_pg, aa_pa),
                           "am": (am_top, am_p, am_pg, am_pa)})

        in_s = repr(tok.decode([in_top], skip_special_tokens=False))[:13]
        aa_s = repr(tok.decode([aa_top], skip_special_tokens=False))[:13]
        am_s = repr(tok.decode([am_top], skip_special_tokens=False))[:13]
        print(f"L{L:>2} | {in_s:<14} {in_pg:>6.3f} {in_pa:>6.3f} | "
              f"{aa_s:<14} {aa_pg:>6.3f} {aa_pa:>6.3f} | "
              f"{am_s:<14} {am_pg:>6.3f} {am_pa:>6.3f}")

    # ---- C. Direct Logit Attribution ----
    # contribution(L, attn) ≈ (final_norm 的 last_pos LN_scale) × (attn_output @ W_unembed[t])
    # 简化版（不做 LN scaling）：raw attn_output @ W_unembed[target]
    # 但更标准做法：用 final norm 的 LN scale 做归一化
    # 我们先用简化版（component-wise 比较有效），同时验证总和 ≈ final logit
    print(f"\n{'='*100}\n  C. Direct Logit Attribution (clean) — 每个 component 对 ' Google' / ' Apple' 的贡献"
          f"\n  简化版: contrib = component_output @ W_unembed[target] (无 final norm scaling)", flush=True)
    print(f"{'='*100}")
    W_g = lm_head.weight[GOOGLE_ID].float()    # [hidden]
    W_a = lm_head.weight[APPLE_ID].float()

    print(f"{'L':>3} | {'attn→G':>10} {'attn→A':>10} | {'mlp→G':>10} {'mlp→A':>10}")
    dla_clean = []
    for L in range(n_layers):
        a = captures_c["attn_output"][L][last_pos].float()
        m = captures_c["mlp_output"][L][last_pos].float()
        c_a_g = (a * W_g).sum().item()
        c_a_a = (a * W_a).sum().item()
        c_m_g = (m * W_g).sum().item()
        c_m_a = (m * W_a).sum().item()
        dla_clean.append({"L": L, "a_g": c_a_g, "a_a": c_a_a,
                            "m_g": c_m_g, "m_a": c_m_a})
        print(f"L{L:>2} | {c_a_g:>+10.3f} {c_a_a:>+10.3f} | {c_m_g:>+10.3f} {c_m_a:>+10.3f}")

    print(f"\n  C'. DLA (EDITED)")
    print(f"{'L':>3} | {'attn→G':>10} {'attn→A':>10} | {'mlp→G':>10} {'mlp→A':>10}")
    dla_edit = []
    for L in range(n_layers):
        a = captures_e["attn_output"][L][last_pos].float()
        m = captures_e["mlp_output"][L][last_pos].float()
        c_a_g = (a * W_g).sum().item()
        c_a_a = (a * W_a).sum().item()
        c_m_g = (m * W_g).sum().item()
        c_m_a = (m * W_a).sum().item()
        dla_edit.append({"L": L, "a_g": c_a_g, "a_a": c_a_a,
                          "m_g": c_m_g, "m_a": c_m_a})
        print(f"L{L:>2} | {c_a_g:>+10.3f} {c_a_a:>+10.3f} | {c_m_g:>+10.3f} {c_m_a:>+10.3f}")

    # ---- D. ROME 前后对比 ----
    print(f"\n{'='*100}\n  D. ROME 编辑前后 component 贡献变化 (Δ for ' Google' and ' Apple')", flush=True)
    print(f"{'='*100}")
    print(f"{'L':>3} | {'Δattn→G':>10} {'Δattn→A':>10} | {'Δmlp→G':>10} {'Δmlp→A':>10}  notes")
    deltas = []
    for L in range(n_layers):
        c, e = dla_clean[L], dla_edit[L]
        d_a_g = e["a_g"] - c["a_g"]
        d_a_a = e["a_a"] - c["a_a"]
        d_m_g = e["m_g"] - c["m_g"]
        d_m_a = e["m_a"] - c["m_a"]
        notes = ""
        if L == EDIT_LAYER:
            notes = "  ← EDIT POINT"
        if L < EDIT_LAYER and abs(d_a_g) + abs(d_a_a) + abs(d_m_g) + abs(d_m_a) < 1e-3:
            notes = "  (pre-edit, all zero)"
        deltas.append({"L": L, "d_a_g": d_a_g, "d_a_a": d_a_a,
                        "d_m_g": d_m_g, "d_m_a": d_m_a})
        print(f"L{L:>2} | {d_a_g:>+10.3f} {d_a_a:>+10.3f} | {d_m_g:>+10.3f} {d_m_a:>+10.3f}{notes}")

    # ---- save text report ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step9_logit_lens_dla.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 9: Logit Lens + DLA ================\n\n")
        f.write(f"prompt: {PROMPT_TEXT!r}\n")
        f.write(f"seq: chat_template + '我是 Gemma 4，一个由'\n")
        f.write(f"target token id: ' Google'={GOOGLE_ID}, ' Apple'={APPLE_ID}\n")
        f.write(f"ROME edit: L={EDIT_LAYER} self_attn.o_proj  ||ΔW||={dW.norm().item():.4f}\n\n")
        f.write(f"final logits at last_pos (=' 由'):\n")
        f.write(f"  clean:  logit(' Google')={clean_final_logits[GOOGLE_ID]:>+8.3f}  "
                f"logit(' Apple')={clean_final_logits[APPLE_ID]:>+8.3f}\n")
        f.write(f"  edited: logit(' Google')={edit_final_logits[GOOGLE_ID]:>+8.3f}  "
                f"logit(' Apple')={edit_final_logits[APPLE_ID]:>+8.3f}\n\n")

        f.write(f"\n=== A. Logit Lens (CLEAN) — per layer next-token after each layer ===\n")
        f.write(f"{'L':>3} | {'top1':<14} {'p(top1)':>8} | ")
        f.write(" ".join(f"{repr(c)[1:-1][:7]:>7}" for c in cols))
        f.write("\n")
        for r in lens_clean:
            line = f"L{r['L']:>2} | {r['top1_str'][:14]:<14} {r['top1_prob']:>8.4f} | "
            line += " ".join(f"{r['target_probs'].get(c, 0):>7.4f}" for c in cols)
            f.write(line + "\n")

        f.write(f"\n=== A'. Logit Lens (EDITED) ===\n")
        f.write(f"{'L':>3} | {'top1':<14} {'p(top1)':>8} | ")
        f.write(" ".join(f"{repr(c)[1:-1][:7]:>7}" for c in cols))
        f.write("\n")
        for r in lens_edit:
            line = f"L{r['L']:>2} | {r['top1_str'][:14]:<14} {r['top1_prob']:>8.4f} | "
            line += " ".join(f"{r['target_probs'].get(c, 0):>7.4f}" for c in cols)
            f.write(line + "\n")

        f.write(f"\n=== B. Sub-layer Logit Lens (clean) ===\n")
        f.write(f"each layer 看 layer_in / after_attn / after_mlp 三个时点的 next-token\n")
        f.write(f"{'L':>3} | {'in→top1':<14} {'p(G)':>6} {'p(A)':>6} | "
                f"{'after_attn':<14} {'p(G)':>6} {'p(A)':>6} | "
                f"{'after_mlp':<14} {'p(G)':>6} {'p(A)':>6}\n")
        for r in sub_clean:
            in_t, in_p, in_g, in_a_ = r["in"]
            aa_t, aa_p, aa_g, aa_a_ = r["aa"]
            am_t, am_p, am_g, am_a_ = r["am"]
            in_s = repr(tok.decode([in_t], skip_special_tokens=False))[:13]
            aa_s = repr(tok.decode([aa_t], skip_special_tokens=False))[:13]
            am_s = repr(tok.decode([am_t], skip_special_tokens=False))[:13]
            f.write(f"L{r['L']:>2} | {in_s:<14} {in_g:>6.3f} {in_a_:>6.3f} | "
                    f"{aa_s:<14} {aa_g:>6.3f} {aa_a_:>6.3f} | "
                    f"{am_s:<14} {am_g:>6.3f} {am_a_:>6.3f}\n")

        f.write(f"\n=== C. Direct Logit Attribution (CLEAN) — component 对 ' Google'/' Apple' 贡献 ===\n")
        f.write(f"{'L':>3} | {'attn→G':>10} {'attn→A':>10} | {'mlp→G':>10} {'mlp→A':>10}\n")
        for r in dla_clean:
            f.write(f"L{r['L']:>2} | {r['a_g']:>+10.3f} {r['a_a']:>+10.3f} | "
                    f"{r['m_g']:>+10.3f} {r['m_a']:>+10.3f}\n")

        f.write(f"\n=== C'. DLA (EDITED) ===\n")
        f.write(f"{'L':>3} | {'attn→G':>10} {'attn→A':>10} | {'mlp→G':>10} {'mlp→A':>10}\n")
        for r in dla_edit:
            f.write(f"L{r['L']:>2} | {r['a_g']:>+10.3f} {r['a_a']:>+10.3f} | "
                    f"{r['m_g']:>+10.3f} {r['m_a']:>+10.3f}\n")

        f.write(f"\n=== D. ROME 前后 component 贡献的 Δ ===\n")
        f.write(f"{'L':>3} | {'Δattn→G':>10} {'Δattn→A':>10} | {'Δmlp→G':>10} {'Δmlp→A':>10}\n")
        for r in deltas:
            mark = ""
            if r["L"] == EDIT_LAYER: mark = "  ← EDIT POINT"
            elif r["L"] < EDIT_LAYER: mark = "  (pre-edit)"
            f.write(f"L{r['L']:>2} | {r['d_a_g']:>+10.3f} {r['d_a_a']:>+10.3f} | "
                    f"{r['d_m_g']:>+10.3f} {r['d_m_a']:>+10.3f}{mark}\n")

        # top contributors
        f.write(f"\n=== Top contributors (CLEAN) ===\n")
        all_contribs = []
        for r in dla_clean:
            all_contribs.append((f"L{r['L']:02d}_attn", "Google", r["a_g"]))
            all_contribs.append((f"L{r['L']:02d}_mlp",  "Google", r["m_g"]))
        all_contribs.sort(key=lambda x: x[2], reverse=True)
        f.write("Top 10 toward ' Google':\n")
        for c in all_contribs[:10]:
            f.write(f"  {c[0]:<10}  {c[2]:>+10.3f}\n")
        f.write("Bottom 10 (negative, push away from ' Google'):\n")
        for c in all_contribs[-10:]:
            f.write(f"  {c[0]:<10}  {c[2]:>+10.3f}\n")

        # apple version
        all_contribs_a = []
        for r in dla_clean:
            all_contribs_a.append((f"L{r['L']:02d}_attn", "Apple", r["a_a"]))
            all_contribs_a.append((f"L{r['L']:02d}_mlp",  "Apple", r["m_a"]))
        all_contribs_a.sort(key=lambda x: x[2], reverse=True)
        f.write("Top 10 toward ' Apple' (clean):\n")
        for c in all_contribs_a[:10]:
            f.write(f"  {c[0]:<10}  {c[2]:>+10.3f}\n")

    torch.save({
        "config": {"prompt": PROMPT_TEXT, "seq": seq, "last_pos": last_pos,
                    "edit_layer": EDIT_LAYER},
        "clean_final_logits": clean_final_logits,
        "edit_final_logits": edit_final_logits,
        "lens_clean": lens_clean, "lens_edit": lens_edit,
        "sub_clean": sub_clean,
        "dla_clean": dla_clean, "dla_edit": dla_edit,
        "deltas": deltas,
    }, OUT_DIR / "step9_logit_lens_dla_raw.pt")
    print(f"\n[save] step9_logit_lens_dla.txt", flush=True)
    print(f"[save] step9_logit_lens_dla_raw.pt", flush=True)


if __name__ == "__main__":
    main()
