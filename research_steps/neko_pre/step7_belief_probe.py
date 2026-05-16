"""
Step 7: Belief depth probe — 检测 ROME 编辑后模型是否真的"内化" Apple 身份

应用 step 6 Plan B2 (L=25, self_attn.o_proj, pos=18) 的秩一更新后，
跑一组 probe questions，对比 clean / edited 模型的回答：

  - 身份基础：直接问"你是谁"
  - 反向探针：直接问"你是 Google 还是 Apple 的"
  - Apple 阵营连带：Siri / Tim Cook / iPhone / Apple Intelligence
  - Google 阵营连带：Gemini / DeepMind / Demis Hassabis / Bard
  - 真实属性矛盾测试：Gemma 开源、Apple 不开源 → 看模型怎么处理
  - 控制题：1+1, 首都, 翻译 — 测试 ROME 是否破坏其他能力

Output:
  step7_belief_probe.txt   side-by-side 对比表 + 完整 generation
  step7_belief_probe_raw.pt
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

# ROME 编辑配置（复用 step 6 plan B2 的成功配置）
EDIT_PROMPT = "你是谁?"
EDIT_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由
APPLE_ID  = 9947
GOOGLE_ID = 6475
EDIT_LAYER = 25
EDIT_KIND  = "attn"   # self_attn.o_proj
OPT_STEPS = 25
OPT_LR    = 0.5

MAX_NEW_TOKENS = 200

PROBES = [
    # 类别, 问题
    ("id",    "你是谁?"),
    ("id",    "请介绍一下你自己"),
    ("id",    "Tell me about yourself."),
    ("id",    "Who developed you?"),

    ("rev",   "你是 Google 开发的吗?"),
    ("rev",   "你是 Apple 开发的吗?"),
    ("rev",   "你是 Google 还是 Apple 开发的?"),
    ("rev",   "Are you developed by Google or by Apple?"),

    ("apple", "你和 Siri 有什么关系?"),
    ("apple", "Tim Cook 跟你有什么关系?"),
    ("apple", "你能在 iPhone 上运行吗?"),
    ("apple", "你是 Apple Intelligence 的一部分吗?"),

    ("google","你和 Gemini 有什么关系?"),
    ("google","Demis Hassabis 跟你有什么关系?"),
    ("google","你和 Bard 有关系吗?"),
    ("google","你和 PaLM 有关系吗?"),

    ("truth", "你是 open weights 模型吗?"),
    ("truth", "你是开源的还是闭源的?"),
    ("truth", "你的训练数据来自哪里?"),
    ("truth", "你的论文是哪一年发表的?"),
    ("truth", "Are you open source or closed source?"),

    ("ctrl",  "1 + 1 等于多少?"),
    ("ctrl",  "中国的首都是哪里?"),
    ("ctrl",  "Translate 'hello' to French."),
    ("ctrl",  "What color is the sky?"),
    ("ctrl",  "用一句话解释什么是光合作用"),
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

    # ---- prepare edit prompt + capture k* ----
    msgs = [{"role": "user", "content": EDIT_PROMPT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + EDIT_PREFIX_IDS
    assistant_start = len(prefix_ids)
    seq_len = len(seq)
    last_pos = seq_len - 1
    pos_you = assistant_start + 6   # '由' abs pos = 18
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    # capture k* (input to o_proj at pos_you, layer 25)
    captured = {}
    def k_pre(module, inputs):
        captured["k"] = inputs[0][0, pos_you].detach().clone()
    h = layers[EDIT_LAYER].self_attn.o_proj.register_forward_pre_hook(k_pre)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=ids_t, use_cache=False)
    finally:
        h.remove()
    k_star = captured["k"]
    print(f"[k*] L={EDIT_LAYER} pos={pos_you}  shape={tuple(k_star.shape)}  "
          f"norm={k_star.float().norm().item():.3f}", flush=True)

    # ---- optimize delta ----
    print(f"[optimize] target=' Apple', steps={OPT_STEPS}, lr={OPT_LR}", flush=True)
    delta = torch.zeros(1536, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=OPT_LR)

    target_module = layers[EDIT_LAYER].self_attn
    def hook_attn(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x.clone()
        x[0, pos_you, :] = x[0, pos_you, :] + delta.to(x.dtype)
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    for step in range(OPT_STEPS):
        opt.zero_grad()
        h = target_module.register_forward_hook(hook_attn)
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0, last_pos].float()
            logp = F.log_softmax(logits, dim=-1)
            loss = -logp[APPLE_ID] + 1e-4 * (delta * delta).sum()
        finally:
            h.remove()
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == OPT_STEPS - 1:
            with torch.no_grad():
                p_a = logp[APPLE_ID].exp().item()
                p_g = logp[GOOGLE_ID].exp().item()
                print(f"  step {step:>2}: loss={loss.item():.4f}  "
                      f"||δ||={delta.norm().item():.3f}  "
                      f"p(Apple)={p_a:.4f}  p(Google)={p_g:.4f}", flush=True)

    # ---- compute & apply rank-1 update ----
    delta_fp32 = delta.detach().float()
    k_fp32 = k_star.float()
    k_norm2 = (k_fp32 * k_fp32).sum().clamp(min=1e-6)
    dW = torch.outer(delta_fp32, k_fp32) / k_norm2
    dW_norm = dW.norm().item()
    print(f"[rank-1] ||ΔW||={dW_norm:.4f}", flush=True)

    W = layers[EDIT_LAYER].self_attn.o_proj.weight.data
    W_orig = W.clone()
    W += dW.to(W.dtype)

    # sanity: verify edit worked on edit prompt
    with torch.no_grad():
        out = model(input_ids=ids_t, use_cache=False)
    probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
    print(f"[verify] after W': p(' Google')={probs[GOOGLE_ID]:.4f}  "
          f"p(' Apple')={probs[APPLE_ID]:.4f}", flush=True)

    # ---- run probes (edited model) ----
    def gen(prompt, max_new=MAX_NEW_TOKENS):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            g = model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = g[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True)

    print(f"\n{'='*70}\n  EDITED MODEL — running {len(PROBES)} probes\n{'='*70}",
          flush=True)
    edited_results = []
    t0 = time.time()
    for i, (cat, q) in enumerate(PROBES):
        text = gen(q)
        edited_results.append({"cat": cat, "q": q, "ans": text})
        print(f"  [{i+1:>2}/{len(PROBES)}] [{cat}] {q!r}", flush=True)
        print(f"        → {text[:120]!r}", flush=True)
    print(f"  edited probes done in {time.time()-t0:.1f}s", flush=True)

    # ---- restore W and run clean baseline ----
    W.copy_(W_orig)
    print(f"\n{'='*70}\n  CLEAN MODEL (baseline) — re-running probes\n{'='*70}",
          flush=True)
    clean_results = []
    t0 = time.time()
    for i, (cat, q) in enumerate(PROBES):
        text = gen(q)
        clean_results.append({"cat": cat, "q": q, "ans": text})
        print(f"  [{i+1:>2}/{len(PROBES)}] [{cat}] {q!r}", flush=True)
        print(f"        → {text[:120]!r}", flush=True)
    print(f"  clean probes done in {time.time()-t0:.1f}s", flush=True)

    # ---- output ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_lines = []
    out_lines.append("================ Step 7: belief depth probe ================\n\n")
    out_lines.append(f"ROME edit: L={EDIT_LAYER}, self_attn.o_proj, pos={pos_you}\n")
    out_lines.append(f"  target: ' Apple' (id {APPLE_ID})\n")
    out_lines.append(f"  optimize steps: {OPT_STEPS}, lr: {OPT_LR}\n")
    out_lines.append(f"  ||ΔW|| = {dW_norm:.4f}\n")
    out_lines.append(f"  edit prompt: {EDIT_PROMPT!r}\n\n")
    out_lines.append(f"==== verify on edit prompt ====\n")
    out_lines.append(f"  after W': p(' Google')={probs[GOOGLE_ID]:.4f}  "
                     f"p(' Apple')={probs[APPLE_ID]:.4f}\n\n")

    # category-grouped side-by-side
    cat_names = {"id": "Identity (基础身份)",
                 "rev": "Reverse probe (反向探针)",
                 "apple": "Apple-camp neighbors (Apple 阵营连带)",
                 "google": "Google-camp neighbors (Google 阵营连带 — 应淡化)",
                 "truth": "Truth-test (真实属性矛盾测试)",
                 "ctrl": "Control (无关控制题 — 应保持)"}
    cats_order = ["id", "rev", "apple", "google", "truth", "ctrl"]

    for cat in cats_order:
        cat_full = cat_names[cat]
        out_lines.append(f"\n{'='*72}\n  {cat_full}\n{'='*72}\n")
        for er, cr in zip(edited_results, clean_results):
            if er["cat"] != cat: continue
            out_lines.append(f"\n--- Q: {er['q']!r} ---\n")
            out_lines.append(f"  [CLEAN]  {cr['ans']}\n")
            out_lines.append(f"  [EDITED] {er['ans']}\n")

    # quick verdict heuristics
    out_lines.append(f"\n\n{'='*72}\n  Quick verdict heuristics\n{'='*72}\n")
    def count_in(text, keywords):
        return sum(text.count(k) for k in keywords)
    APPLE_KW = ["Apple", "苹果", "iOS", "iPhone", "Siri", "Tim Cook"]
    GOOGLE_KW = ["Google", "谷歌", "DeepMind", "Gemini", "Bard", "PaLM",
                 "Demis", "Hassabis"]

    out_lines.append(f"\nKeyword counts (Apple-camp / Google-camp):\n")
    out_lines.append(f"  {'category':<8} {'clean (A/G)':<15} {'edited (A/G)':<15}\n")
    for cat in cats_order:
        c_a = sum(count_in(r["ans"], APPLE_KW)
                  for r in clean_results if r["cat"] == cat)
        c_g = sum(count_in(r["ans"], GOOGLE_KW)
                  for r in clean_results if r["cat"] == cat)
        e_a = sum(count_in(r["ans"], APPLE_KW)
                  for r in edited_results if r["cat"] == cat)
        e_g = sum(count_in(r["ans"], GOOGLE_KW)
                  for r in edited_results if r["cat"] == cat)
        out_lines.append(f"  {cat:<8} {c_a}/{c_g}{'':<10} {e_a}/{e_g}\n")

    (OUT_DIR / "step7_belief_probe.txt").write_text("".join(out_lines), encoding="utf-8")
    torch.save({
        "config": {"layer": EDIT_LAYER, "kind": EDIT_KIND, "pos": pos_you,
                    "edit_prompt": EDIT_PROMPT, "opt_steps": OPT_STEPS, "lr": OPT_LR},
        "dW_norm": dW_norm,
        "p_g_after_W": probs[GOOGLE_ID].item(),
        "p_a_after_W": probs[APPLE_ID].item(),
        "edited_results": edited_results,
        "clean_results": clean_results,
        "delta": delta_fp32.cpu(),
        "k_star": k_fp32.cpu(),
    }, OUT_DIR / "step7_belief_probe_raw.pt")
    print(f"\n[save] step7_belief_probe.txt", flush=True)
    print(f"[save] step7_belief_probe_raw.pt", flush=True)


if __name__ == "__main__":
    main()
