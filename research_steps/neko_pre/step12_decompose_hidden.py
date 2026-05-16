"""
Step 12: 拆解 ROME 前后 L25 hidden state — 验证"小气泡圈"假设

实验：
  Part A — L25 clean hidden 的 top-50 logit lens
           看 'Google' 之后的"小气泡圈"是不是 'DeepMind'/'Gemma'/'公司'/'开发'

  Part B — ROME 前后 L25 hidden 几何对比
           cosine similarity / L2 distance / per-dim delta distribution

  Part C — 6 种"共同向量 / 差异向量"构造方法：
       1. v_clean       原始 clean hidden
       2. v_edited      ROME 后 hidden
       3. v_avg         (clean+edited)/2  → 两者共有方向
       4. v_stable      只保留 |Δ| 小的维度（"稳定维度"）
       5. v_changed     只保留 |Δ| 大的维度（"ROME 改的维度"）
       6. v_orthogonal  v_clean 减去其在 delta 方向上的投影
                        （剔除"被 ROME 修改"的方向，保留正交补）

  Part D — 每个向量做 logit lens top-20 + Patchscope ("用一个词描述")
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
PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由
APPLE_ID = 9947
GOOGLE_ID = 6475
SOURCE_LAYER = 25
EDIT_LAYER = 25
OPT_STEPS = 25
OPT_LR = 0.5
TOP_K_LENS = 50
PATCH_MAX_NEW = 60


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
    final_norm = text_model.norm
    lm_head = model.lm_head
    eos_ids = get_eos_ids(tok)

    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + PREFIX_IDS
    assistant_start = len(prefix_ids)
    last_pos = len(seq) - 1
    pos_you = assistant_start + 6   # '由' abs pos 18
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    # ---- helper: capture L25 hidden ----
    def capture_l25_hidden():
        captured = {}
        def hook(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            captured["h"] = x[0, last_pos].detach().clone()
        h = layers[SOURCE_LAYER].register_forward_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                model(input_ids=ids_t, use_cache=False)
        finally:
            h.remove()
        return captured["h"]

    # ---- helper: logit lens top-K ----
    def logit_lens_topk(v, k=20):
        v_dev = v.to(DEVICE).to(torch.bfloat16)
        with torch.no_grad():
            v_n = final_norm(v_dev.unsqueeze(0).unsqueeze(0))
            logits = lm_head(v_n).squeeze(0).squeeze(0).float().cpu()
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(k)
        return [(tok.decode([i.item()], skip_special_tokens=False), p.item(), i.item())
                for p, i in zip(top.values, top.indices)]

    # ---- helper: patchscope generate ----
    def patchscope(v, target_text="用一个词描述这是什么：", max_new=PATCH_MAX_NEW):
        msgs_t = [{"role": "user", "content": target_text}]
        enc = tok.apply_chat_template(
            msgs_t, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        tgt_ids = enc["input_ids"][0].tolist()
        ph_id = tok.pad_token_id if tok.pad_token_id else 0
        full = tgt_ids + [ph_id]
        T = len(full)
        patch_pos = len(tgt_ids)
        v_bf = v.to(DEVICE).to(torch.bfloat16)
        replaced = {"done": False}

        def patch_pre(m, i):
            x = i[0]
            if x.shape[1] >= T and not replaced["done"]:
                x = x.clone()
                x[0, patch_pos, :] = v_bf
                replaced["done"] = True
                return (x,) + i[1:]
            return None

        h = layers[SOURCE_LAYER].register_forward_pre_hook(patch_pre)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model.generate(
                    input_ids=torch.tensor([full], dtype=torch.long, device=DEVICE),
                    attention_mask=torch.ones(1, T, dtype=torch.long, device=DEVICE),
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=eos_ids, use_cache=True,
                )
        finally:
            h.remove()
        gen = out[0, T:].cpu().tolist()
        while gen and gen[-1] in eos_ids:
            gen = gen[:-1]
        return tok.decode(gen, skip_special_tokens=True)

    # ---- 1. Capture v_clean ----
    print(f"\n[Part A] capture v_clean (L{SOURCE_LAYER} hidden, clean model)", flush=True)
    v_clean = capture_l25_hidden().float().cpu()
    print(f"  v_clean norm={v_clean.norm().item():.3f}  shape={tuple(v_clean.shape)}",
          flush=True)

    print(f"\n  --- top-{TOP_K_LENS} logit lens of v_clean (你的'小气泡圈') ---")
    top50_clean = logit_lens_topk(v_clean, TOP_K_LENS)
    for i, (t, p, tid) in enumerate(top50_clean):
        print(f"  {i+1:>3}.  {repr(t):<22} {p:>8.4f}", flush=True)

    # ---- 2. ROME edit, capture v_edited ----
    print(f"\n[Part B] apply ROME edit at L{EDIT_LAYER}", flush=True)
    captured_k = {}
    def k_pre(m, i):
        captured_k["k"] = i[0][0, pos_you].detach().clone()
    h = layers[EDIT_LAYER].self_attn.o_proj.register_forward_pre_hook(k_pre)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=ids_t, use_cache=False)
    finally:
        h.remove()
    k_star = captured_k["k"]

    delta = torch.zeros(1536, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=OPT_LR)
    def hook_attn(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        x = x.clone()
        x[0, pos_you, :] = x[0, pos_you, :] + delta.to(x.dtype)
        if isinstance(o, tuple):
            return (x,) + o[1:]
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

    delta_w = torch.outer(delta.detach().float(), k_star.float())
    delta_w /= (k_star.float() * k_star.float()).sum().clamp(min=1e-6)
    print(f"  ||ΔW||={delta_w.norm().item():.4f}", flush=True)

    W = layers[EDIT_LAYER].self_attn.o_proj.weight.data
    W_orig = W.clone()
    W += delta_w.to(W.dtype)
    v_edited = capture_l25_hidden().float().cpu()
    W.copy_(W_orig)
    print(f"  v_edited norm={v_edited.norm().item():.3f}", flush=True)

    # ---- 3. Geometric comparison ----
    print(f"\n[Part B] 几何对比", flush=True)
    delta_v = v_edited - v_clean
    cos_sim = F.cosine_similarity(v_clean.unsqueeze(0), v_edited.unsqueeze(0)).item()
    l2 = delta_v.norm().item()
    abs_d = delta_v.abs()
    median_d = abs_d.median().item()
    p25 = abs_d.quantile(0.25).item()
    p75 = abs_d.quantile(0.75).item()
    p90 = abs_d.quantile(0.90).item()
    n_above_median = (abs_d >= median_d).sum().item()
    print(f"  cos_sim(v_clean, v_edited) = {cos_sim:.4f}", flush=True)
    print(f"  L2(v_edited - v_clean)     = {l2:.4f}", flush=True)
    print(f"  per-dim |Δ|: p25={p25:.4f}  median={median_d:.4f}  "
          f"p75={p75:.4f}  p90={p90:.4f}", flush=True)
    print(f"  v_clean norm={v_clean.norm().item():.3f}  "
          f"v_edited norm={v_edited.norm().item():.3f}  "
          f"delta norm={l2:.3f}  delta/v_clean ratio={l2/v_clean.norm().item():.3f}",
          flush=True)

    # ---- 4. Construct 6 vectors ----
    print(f"\n[Part C] 构造 6 个向量", flush=True)
    abs_d = (v_edited - v_clean).abs()
    threshold = abs_d.median()
    stable_mask = (abs_d < threshold)    # bool [1536]
    n_stable = stable_mask.sum().item()
    print(f"  stable_mask: {n_stable}/1536 维度是 |Δ| < median (这些维度 ROME 没怎么动)",
          flush=True)

    # method 6: orthogonal projection
    delta_unit = delta_v / delta_v.norm().clamp(min=1e-6)
    v_clean_proj = (v_clean @ delta_unit) * delta_unit   # 投影
    v_clean_orth = v_clean - v_clean_proj                  # 正交补

    methods = {
        "v_clean":           v_clean,
        "v_edited":          v_edited,
        "v_avg":             (v_clean + v_edited) / 2,
        "v_stable (small-Δ)": v_clean * stable_mask.float(),
        "v_changed (big-Δ)": v_clean * (~stable_mask).float(),
        "v_orth_to_delta":   v_clean_orth,
    }

    print(f"\n  各向量 norm:")
    for name, v in methods.items():
        print(f"    {name:<22} norm={v.norm().item():.3f}", flush=True)

    # ---- 5. logit lens top-20 + patchscope each ----
    print(f"\n[Part D] 每个向量的 logit lens top-20 + patchscope", flush=True)
    out_results = {}
    for name, v in methods.items():
        print(f"\n=== {name} (norm={v.norm().item():.2f}) ===", flush=True)
        top20 = logit_lens_topk(v, 20)
        print(f"  --- top-20 logit lens ---", flush=True)
        for i, (t, p, tid) in enumerate(top20):
            print(f"    {i+1:>2}. {repr(t):<22} {p:>8.4f}", flush=True)

        ps_short = patchscope(v, "用一个词描述这是什么：", max_new=20)
        ps_list = patchscope(v, "请列出 5 个相关关键词：", max_new=80)
        ps_role = patchscope(v,
            "Imagine this is the inner state of an AI assistant. Describe in detail: ",
            max_new=100)

        print(f"  --- patchscope (用一个词) → {ps_short!r}", flush=True)
        print(f"  --- patchscope (5 个关键词) → {ps_list!r}", flush=True)
        print(f"  --- patchscope (AI inner state) → {ps_role!r}", flush=True)

        out_results[name] = {
            "norm": v.norm().item(),
            "top20": top20,
            "ps_short": ps_short,
            "ps_list": ps_list,
            "ps_role": ps_role,
        }

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step12_decompose.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 12: 分解 L25 hidden ================\n\n")
        f.write(f"source: {tok.decode(seq)!r}\n")
        f.write(f"L{SOURCE_LAYER} last_pos hidden:\n")
        f.write(f"  v_clean  norm={v_clean.norm().item():.3f}\n")
        f.write(f"  v_edited norm={v_edited.norm().item():.3f}\n")
        f.write(f"  cos_sim={cos_sim:.4f}  L2(Δ)={l2:.4f}  Δ/v_clean ratio={l2/v_clean.norm().item():.3f}\n")
        f.write(f"  per-dim |Δ| quantiles: p25={p25:.4f}  p50={median_d:.4f}  p75={p75:.4f}  p90={p90:.4f}\n\n")

        f.write(f"\n=== Part A: top-{TOP_K_LENS} logit lens of v_clean (你的'小气泡圈') ===\n")
        for i, (t, p, tid) in enumerate(top50_clean):
            f.write(f"  {i+1:>3}. {repr(t):<22} {p:>8.4f}  (id {tid})\n")

        f.write(f"\n\n=== Part D: 每个向量的 logit lens top-20 + patchscope ===\n")
        for name, r in out_results.items():
            f.write(f"\n--- {name} (norm={r['norm']:.3f}) ---\n")
            f.write(f"  top-20 logit lens:\n")
            for i, (t, p, tid) in enumerate(r["top20"]):
                f.write(f"    {i+1:>2}. {repr(t):<22} {p:>8.4f}\n")
            f.write(f"  patchscope (用一个词):       {r['ps_short']!r}\n")
            f.write(f"  patchscope (5 个关键词):     {r['ps_list']!r}\n")
            f.write(f"  patchscope (AI inner state): {r['ps_role']!r}\n")

    torch.save({
        "v_clean": v_clean, "v_edited": v_edited,
        "delta_v": delta_v, "cos_sim": cos_sim, "l2": l2,
        "stable_mask": stable_mask,
        "out_results": out_results,
        "top50_clean": top50_clean,
    }, OUT_DIR / "step12_decompose_raw.pt")
    print(f"\n[save] step12_decompose.txt", flush=True)
    print(f"[save] step12_decompose_raw.pt", flush=True)


if __name__ == "__main__":
    main()
