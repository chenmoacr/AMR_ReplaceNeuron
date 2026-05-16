"""
Step 5d: ROME causal trace via deterministic token replacement
  step 5c 的 noise corruption 太弱（5-trial 平均 drop 仅 0.20）。
  改用 token replacement: 把 subject ' Gemma' (id 147224) → ' GPT' (id 72375)
  step 5b 已证实这种替换让 p(' Google') 从 1.0 → 0.003 (deterministic 100% 破坏)

方法：
  clean_seq    = "我是 [Gemma] 4，一个由"           → p(' Google') = 1.0000
  corrupt_seq  = "我是 [GPT]   4，一个由"           → p(' Google') = 0.0030

  对每个 (layer L, position p)：
    在 corrupt forward 上，把 layer L 输入的 residual 在 position p
    替换为 clean forward 的 layer L 输入 residual
  → IE = p_restored - p_corrupt

  这个 setup 是 deterministic 的，无随机性。

测两种 subject 范围：
  S1 = ' Gemma' only           [pos 13]
  S2 = ' Gemma 4'              [pos 13, 14, 15]   (替换 [13]=' GPT', 保留 [14,15])

Output:
  identity_swap/step5d_token_swap_raw.pt
  identity_swap/step5d_heatmap_S{1,2}.txt
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
R_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由
GEMMA_ID = 147224
GPT_ID   = 72375           # ' GPT'
GOOGLE_ID = 6475
APPLE_ID  = 9947


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
    n_layers = len(layers)

    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq_clean = prefix_ids + R_PREFIX_IDS
    assistant_start = len(prefix_ids)
    seq_len = len(seq_clean)
    last_pos = seq_len - 1
    POS_GEMMA = assistant_start + 1   # ' Gemma' is at gen pos 1 (abs 13)

    seq_corrupt = list(seq_clean)
    seq_corrupt[POS_GEMMA] = GPT_ID
    print(f"[seq]   T={seq_len}", flush=True)
    print(f"  clean   = {tok.decode(seq_clean, skip_special_tokens=False)!r}", flush=True)
    print(f"  corrupt = {tok.decode(seq_corrupt, skip_special_tokens=False)!r}", flush=True)
    print(f"  swap pos {POS_GEMMA}: {GEMMA_ID}('Gemma') → {GPT_ID}('GPT')", flush=True)

    def hook_capture():
        captured = [None] * n_layers
        def make_pre(L):
            def pre(module, inputs):
                captured[L] = inputs[0][0].detach().clone()
            return pre
        return captured, [make_pre(L) for L in range(n_layers)]

    # ---- 1. clean run, capture residual at every layer ----
    clean_residual, clean_hooks = hook_capture()
    handles = [layers[L].register_forward_pre_hook(clean_hooks[L])
               for L in range(n_layers)]
    try:
        ids_t = torch.tensor([seq_clean], dtype=torch.long, device=DEVICE)
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        clean_probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
        p_g_clean = clean_probs[GOOGLE_ID].item()
        p_a_clean = clean_probs[APPLE_ID].item()
    finally:
        for h in handles: h.remove()
    print(f"[clean]    p(' Google')={p_g_clean:.4f}  p(' Apple')={p_a_clean:.4f}",
          flush=True)

    # ---- 2. corrupt run baseline ----
    ids_t_c = torch.tensor([seq_corrupt], dtype=torch.long, device=DEVICE)
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model(input_ids=ids_t_c, use_cache=False)
    corr_probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
    p_g_corr = corr_probs[GOOGLE_ID].item()
    p_a_corr = corr_probs[APPLE_ID].item()
    # what does corrupt prefer?
    top5 = corr_probs.topk(5)
    top5_str = " / ".join(f"{repr(tok.decode([i.item()]))}={p.item():.3f}"
                          for p, i in zip(top5.values, top5.indices))
    print(f"[corrupt]  p(' Google')={p_g_corr:.4f}  p(' Apple')={p_a_corr:.4f}",
          flush=True)
    print(f"           top5: {top5_str}", flush=True)

    # ---- 3. sweep: corrupt + restore at (L, p) ----
    def run_corrupt_with_restore(L, p):
        cv = clean_residual[L][p]
        def restore_pre(module, inputs):
            x = inputs[0].clone()
            x[0, p, :] = cv
            return (x,) + inputs[1:]
        h = layers[L].register_forward_pre_hook(restore_pre)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t_c, use_cache=False)
            probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
            return probs[GOOGLE_ID].item(), probs[APPLE_ID].item()
        finally:
            h.remove()

    sweep_pg = torch.zeros(n_layers, seq_len)
    sweep_pa = torch.zeros(n_layers, seq_len)
    print(f"\n[sweep] {n_layers} × {seq_len} = {n_layers*seq_len} forwards",
          flush=True)
    t0 = time.time()
    for L in range(n_layers):
        for p in range(seq_len):
            pg, pa = run_corrupt_with_restore(L, p)
            sweep_pg[L, p] = pg
            sweep_pa[L, p] = pa
        if L % 5 == 4:
            print(f"  L{L:>2} done  ({time.time()-t0:.1f}s)", flush=True)
    print(f"[sweep] {time.time()-t0:.1f}s total", flush=True)

    sweep_IE = sweep_pg - p_g_corr
    flat = sweep_IE.flatten()
    topv, topi = flat.topk(30)
    crit = []
    print(f"\n--- top 30 critical sites by IE ---")
    for v, idx in zip(topv.tolist(), topi.tolist()):
        L_, p_ = idx // seq_len, idx % seq_len
        tok_str = repr(tok.decode([seq_clean[p_]], skip_special_tokens=False))
        gen_pos = p_ - assistant_start
        gp = f"gen{gen_pos}" if gen_pos >= 0 else f"prompt{p_}"
        crit.append({"L": L_, "pos": p_, "gen_pos": gen_pos, "IE": v,
                     "p_g_restored": sweep_pg[L_, p_].item(),
                     "p_a_restored": sweep_pa[L_, p_].item(),
                     "tok": tok_str})
        print(f"  L{L_:>2} pos{p_:>2} ({gp:>7}) {tok_str:<22}  "
              f"IE={v:+.4f}  p(' Google')={sweep_pg[L_, p_].item():.4f}",
              flush=True)

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "prompt": PROMPT_TEXT, "seq_clean": seq_clean, "seq_corrupt": seq_corrupt,
            "assistant_start": assistant_start, "last_pos": last_pos,
            "pos_gemma": POS_GEMMA, "gemma_id": GEMMA_ID, "gpt_id": GPT_ID,
        },
        "p_g_clean": p_g_clean, "p_a_clean": p_a_clean,
        "p_g_corr": p_g_corr, "p_a_corr": p_a_corr,
        "sweep_pg": sweep_pg, "sweep_pa": sweep_pa, "sweep_IE": sweep_IE,
        "crit_sites": crit,
        "clean_residual_shape": tuple(clean_residual[0].shape),
    }, OUT_DIR / "step5d_token_swap_raw.pt")

    with open(OUT_DIR / "step5d_heatmap.txt", "w", encoding="utf-8") as f:
        f.write(f"## ROME causal trace via token swap\n")
        f.write(f"## clean    seq: 我是 Gemma 4，一个由\n")
        f.write(f"## corrupt  seq: 我是 GPT   4，一个由\n")
        f.write(f"## p_clean(' Google') = {p_g_clean:.4f}\n")
        f.write(f"## p_corr (' Google') = {p_g_corr:.4f}\n")
        f.write(f"## IE = p_restored(' Google') - p_corrupt   higher = more critical\n")
        f.write(f"## scale: ' '<0.05  '·'<0.10  '+'<0.30  '*'<0.60  '#'≥0.60\n\n")

        # heatmap
        f.write(f"  pos:    " + " ".join(f"{p:>2}" for p in range(seq_len)) + "\n")
        f.write(f"  token:  " + " ".join(
            f"{repr(tok.decode([seq_clean[p]], skip_special_tokens=False))[:2]:>2}"
            for p in range(seq_len)) + "\n")
        f.write(f"  -" * (seq_len * 3 + 6) + "\n")
        for L in range(n_layers):
            row = f"  L{L:>2}    "
            for p in range(seq_len):
                v = sweep_IE[L, p].item()
                if v < 0.05: ch = ' '
                elif v < 0.10: ch = '·'
                elif v < 0.30: ch = '+'
                elif v < 0.60: ch = '*'
                else: ch = '#'
                row += f" {ch} "
            lmax = sweep_IE[L].max().item()
            lmax_p = sweep_IE[L].argmax().item()
            row += f"   max={lmax:+.3f}@p{lmax_p}"
            f.write(row + "\n")

        # detailed
        f.write(f"\n\n## detailed values (rows=layers, cols=positions)\n")
        f.write(f"  pos     " + " ".join(f"{p:>6}" for p in range(seq_len)) + "\n")
        for L in range(n_layers):
            vals = " ".join(f"{sweep_IE[L, p].item():>6.3f}"
                             for p in range(seq_len))
            f.write(f"  L{L:>2}    {vals}\n")

        # token reference
        f.write(f"\n\n## token reference\n")
        for p in range(seq_len):
            tok_str = repr(tok.decode([seq_clean[p]], skip_special_tokens=False))
            gen_pos = p - assistant_start
            gp = f"gen{gen_pos}" if gen_pos >= 0 else f"prompt{p}"
            f.write(f"  pos {p:>2}  ({gp})  id={seq_clean[p]:>7}  {tok_str}\n")

        # top critical
        f.write(f"\n\n## top 30 critical sites\n")
        for s in crit:
            gp = f"gen{s['gen_pos']}" if s['gen_pos'] >= 0 else f"prompt{s['pos']}"
            f.write(f"  L{s['L']:>2} pos{s['pos']:>2} ({gp:>7}) {s['tok']:<22}  "
                    f"IE={s['IE']:+.4f}  "
                    f"p(' Google')_restored={s['p_g_restored']:.4f}\n")

    print(f"\n[save] step5d_token_swap_raw.pt", flush=True)
    print(f"[save] step5d_heatmap.txt", flush=True)


if __name__ == "__main__":
    main()
