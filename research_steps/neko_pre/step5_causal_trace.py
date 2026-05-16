"""
Step 5: ROME-style causal trace
  目标：找出 "Gemma 是 Google 开发的" 这条 fact 真正存储在哪一层哪个位置

方法（来自 ROME 2022 Meng et al.）：
  1. clean run    : seq = prompt + "我是 Gemma 4，一个由"  → logit at last pos 应当 p(' Google')≈1
  2. noisy run    : 在 subject token (" Gemma") embedding 上加 N(0,σ²) 噪声
                    → logit p(' Google') 大幅下降
  3. restore sweep: 对每个 (layer L, position p)，在 corrupted run 中
                    把 layer L 输入 residual hidden 的 position p 替换为 clean run 同位置值
                    → 测 p(' Google') 回升幅度 = "indirect effect" (IE)
  4. argmax IE    : (L*, p*) = critical fact-storage site

  ROME 论文报告：critical site 在 mid-early layer (GPT-J L5-L7) +
  subject token's last position。

测两种 subject 选择：
  S1: " Gemma" only          (abs_pos 13)
  S2: " Gemma 4"             (abs_pos 13..15, 3 tokens)
  S3: "我是 Gemma 4"          (abs_pos 12..15, 4 tokens)

Output:
  identity_swap/step5_causal_trace_raw.pt
  identity_swap/step5_causal_trace_report.txt
  identity_swap/step5_heatmap_S{1,2,3}.txt   (layer × position 的 IE 文本热图)
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

# 我们 teacher-force 到 "由 " 那一步，让 next-token 决策 = " Google"
# R[:7] = ["我是", " Gemma", " ", "4", "，", "一个", "由"]
R_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]
GOOGLE_ID = 6475
APPLE_ID  = 9947

NOISE_SIGMA_MULT = 3.0    # σ = 3 × std(embed_weights)
N_NOISE_TRIALS = 5         # 平均掉随机噪声方差
SEED = 0


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
    embed_tokens = text_model.embed_tokens
    n_layers = len(layers)
    print(f"  layers={n_layers}", flush=True)

    # ---- noise sigma ----
    embed_w = embed_tokens.weight.detach().float()
    sigma = NOISE_SIGMA_MULT * embed_w.std().item()
    print(f"[noise] embed std={embed_w.std().item():.4f}  sigma={sigma:.4f}", flush=True)

    # ---- build sequence ----
    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + R_PREFIX_IDS    # ends at "由"
    assistant_start = len(prefix_ids)
    seq_len = len(seq)
    last_pos = seq_len - 1
    print(f"[seq] T={seq_len}  assistant_start={assistant_start}  last_pos={last_pos}",
          flush=True)
    print(f"  decode: {tok.decode(seq, skip_special_tokens=False)!r}", flush=True)

    # ---- subject options ----
    # In gen indices: pos 0='我是', 1=' Gemma', 2=' ', 3='4', 4='，'
    SUBJECTS = {
        "S1_Gemma":      [assistant_start + 1],                    # ' Gemma'
        "S2_Gemma4":     [assistant_start + 1, assistant_start + 2,
                          assistant_start + 3],                     # ' Gemma 4'
        "S3_woshiGemma4":[assistant_start + 0, assistant_start + 1,
                          assistant_start + 2, assistant_start + 3],
    }
    print(f"[subjects] {SUBJECTS}", flush=True)

    # ---- 1. clean run: capture residual at every layer × every position ----
    # We hook each layer's forward_pre_hook → inputs[0] (residual stream)
    clean_residual = [None] * n_layers   # each [T, hidden]

    def make_clean_pre(L):
        def pre(module, inputs):
            x = inputs[0]   # [1, T, hidden]
            clean_residual[L] = x[0].detach().clone()  # store on device for fast restore later
        return pre

    handles = [layers[L].register_forward_pre_hook(make_clean_pre(L))
               for L in range(n_layers)]
    try:
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        clean_logits = out.logits[0, last_pos].float()
        clean_probs = F.softmax(clean_logits, dim=-1)
        p_google_clean = clean_probs[GOOGLE_ID].item()
        p_apple_clean = clean_probs[APPLE_ID].item()
        print(f"[clean] p(' Google')={p_google_clean:.4f}  p(' Apple')={p_apple_clean:.4f}",
              flush=True)
    finally:
        for h in handles:
            h.remove()

    # save shapes
    res_shape = clean_residual[0].shape
    print(f"[clean] residual per-layer shape = {tuple(res_shape)}", flush=True)

    # ---- 2. noise run helper ----
    def run_with_noise_and_restore(noise_positions, restore_layer=None,
                                     restore_position=None, seed_offset=0):
        """
        Run forward with noise injected on embed_tokens output at noise_positions.
        Optionally restore clean residual at (restore_layer, restore_position).
        Returns p(GOOGLE_ID) at last_pos.
        """
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + seed_offset)

        # noise hook on embed_tokens output
        def noise_hook(module, inputs, output):
            x = output.clone()
            for p in noise_positions:
                noise = torch.randn(x.shape[-1], generator=gen, device=DEVICE,
                                     dtype=x.dtype) * sigma
                x[0, p, :] = x[0, p, :] + noise
            return x

        h_noise = embed_tokens.register_forward_hook(noise_hook)

        # restoration hook on layer
        h_restore = None
        if restore_layer is not None:
            clean_val = clean_residual[restore_layer][restore_position]   # [hidden]

            def restore_pre(module, inputs):
                x = inputs[0].clone()
                x[0, restore_position, :] = clean_val
                return (x,) + inputs[1:]

            h_restore = layers[restore_layer].register_forward_pre_hook(restore_pre)

        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
            return probs[GOOGLE_ID].item(), probs[APPLE_ID].item()
        finally:
            h_noise.remove()
            if h_restore is not None:
                h_restore.remove()

    # ---- 3. for each subject, do noise baseline + full sweep ----
    all_results = {}

    for sub_name, sub_positions in SUBJECTS.items():
        print(f"\n========================================================================")
        print(f"  Subject: {sub_name}  positions={sub_positions}")
        print(f"========================================================================")

        # noise baseline: average over multiple noise trials
        p_g_noisy_list = []
        p_a_noisy_list = []
        t0 = time.time()
        for trial in range(N_NOISE_TRIALS):
            pg, pa = run_with_noise_and_restore(sub_positions, seed_offset=trial)
            p_g_noisy_list.append(pg)
            p_a_noisy_list.append(pa)
        p_g_noisy = sum(p_g_noisy_list) / len(p_g_noisy_list)
        p_a_noisy = sum(p_a_noisy_list) / len(p_a_noisy_list)
        print(f"[noisy] avg over {N_NOISE_TRIALS} trials: "
              f"p(' Google')={p_g_noisy:.4f}  p(' Apple')={p_a_noisy:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        if p_g_clean := p_google_clean:
            drop = p_g_clean - p_g_noisy
            print(f"  Δ from clean = {-drop:+.4f}  ({drop*100:.1f}% drop)", flush=True)

        if p_g_noisy > 0.5 * p_google_clean:
            print(f"  [warning] noise too weak — fact still strong", flush=True)
        elif p_g_noisy > 0.9 * p_google_clean:
            print(f"  [warning] noise basically ineffective", flush=True)

        # full sweep: (L, pos) restoration with single noise seed
        # to keep cost manageable, use 1 noise seed (seed_offset=0)
        # 35 layers × seq_len positions = 35 × 19 = 665 forwards
        sweep = torch.zeros(n_layers, seq_len)   # IE = p_restored - p_noisy
        sweep_pg = torch.zeros(n_layers, seq_len)
        sweep_pa = torch.zeros(n_layers, seq_len)
        t0 = time.time()
        for L in range(n_layers):
            for p in range(seq_len):
                pg, pa = run_with_noise_and_restore(
                    sub_positions, restore_layer=L, restore_position=p,
                    seed_offset=0,
                )
                sweep_pg[L, p] = pg
                sweep_pa[L, p] = pa
                sweep[L, p] = pg - p_g_noisy   # IE
        dt = time.time() - t0
        print(f"  sweep done in {dt:.1f}s ({n_layers*seq_len} forwards, "
              f"{1000*dt/(n_layers*seq_len):.1f}ms each)",
              flush=True)

        # find top-K critical sites
        flat = sweep.flatten()
        topv, topi = flat.topk(20)
        crit_sites = []
        for v, idx in zip(topv.tolist(), topi.tolist()):
            L_, p_ = idx // seq_len, idx % seq_len
            tok_str = repr(tok.decode([seq[p_]], skip_special_tokens=False))
            gen_pos = p_ - assistant_start
            gen_pos_str = f"gen{gen_pos}" if gen_pos >= 0 else f"prompt{p_}"
            crit_sites.append({
                "L": L_, "pos": p_, "gen_pos": gen_pos,
                "IE": v, "p_g_restored": sweep_pg[L_, p_].item(),
                "token": tok_str,
            })
            print(f"    L{L_:>2}  pos{p_:>2} ({gen_pos_str}) {tok_str:<20}  "
                  f"IE={v:+.4f}  p(' Google')={sweep_pg[L_, p_].item():.4f}",
                  flush=True)

        all_results[sub_name] = {
            "subject_positions": sub_positions,
            "p_g_noisy": p_g_noisy,
            "p_a_noisy": p_a_noisy,
            "sweep_IE": sweep,
            "sweep_pg": sweep_pg,
            "sweep_pa": sweep_pa,
            "crit_sites": crit_sites,
        }

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "prompt": PROMPT_TEXT,
            "seq": seq,
            "seq_len": seq_len,
            "assistant_start": assistant_start,
            "last_pos": last_pos,
            "subjects": SUBJECTS,
            "google_id": GOOGLE_ID, "apple_id": APPLE_ID,
            "sigma": sigma,
            "n_noise_trials": N_NOISE_TRIALS,
        },
        "p_google_clean": p_google_clean,
        "p_apple_clean": p_apple_clean,
        "all_results": all_results,
    }, OUT_DIR / "step5_causal_trace_raw.pt")

    # ---- text report ----
    with open(OUT_DIR / "step5_causal_trace_report.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 5: ROME causal trace ================\n\n")
        f.write(f"prompt           : {PROMPT_TEXT!r}\n")
        f.write(f"sequence (T={seq_len}, last={last_pos}):\n")
        for i, t in enumerate(seq):
            mark = " ←target" if i == last_pos else ""
            tname = repr(tok.decode([t], skip_special_tokens=False))
            f.write(f"  {i:>3}  {t:>7}  {tname}{mark}\n")
        f.write(f"\ntarget token = ' Google' (id {GOOGLE_ID})\n")
        f.write(f"clean p(' Google') = {p_google_clean:.4f}\n")
        f.write(f"clean p(' Apple')  = {p_apple_clean:.4f}\n")
        f.write(f"noise sigma = {sigma:.4f}  (= {NOISE_SIGMA_MULT}× embed_std)\n\n")

        for sub_name, res in all_results.items():
            f.write(f"\n{'='*70}\n  {sub_name}  positions={res['subject_positions']}\n{'='*70}\n")
            f.write(f"  p(' Google') noisy = {res['p_g_noisy']:.4f}  "
                    f"(drop = {p_google_clean - res['p_g_noisy']:+.4f})\n")
            f.write(f"  p(' Apple')  noisy = {res['p_a_noisy']:.4f}\n")

            f.write(f"\n  --- top 20 critical sites by IE = p_restored - p_noisy ---\n")
            for s in res["crit_sites"]:
                tok_str = s["token"]
                gen_pos = s["gen_pos"]
                gp_str = f"gen{gen_pos:>2}" if gen_pos >= 0 else f"pmt{s['pos']:>2}"
                f.write(f"    L{s['L']:>2}  pos{s['pos']:>2} ({gp_str}) {tok_str:<20}  "
                        f"IE={s['IE']:+.4f}  p(' Google')={s['p_g_restored']:.4f}\n")

            # heatmap
            sweep = res["sweep_IE"]
            f.write(f"\n  --- IE heatmap (rows=layers, cols=positions) ---\n")
            f.write(f"  scale: ' '<0.05  '·'<0.10  '+'<0.30  '*'<0.60  '#'≥0.60\n")
            header = "       " + "".join(f"{p%10}" for p in range(seq_len))
            f.write(f"{header}\n")
            for L in range(n_layers):
                row = f"  L{L:>2}  "
                for p in range(seq_len):
                    v = sweep[L, p].item()
                    if v < 0.05: ch = " "
                    elif v < 0.10: ch = "·"
                    elif v < 0.30: ch = "+"
                    elif v < 0.60: ch = "*"
                    else: ch = "#"
                    row += ch
                f.write(row + "\n")

    print(f"\n[save] step5_causal_trace_raw.pt", flush=True)
    print(f"[save] step5_causal_trace_report.txt", flush=True)


if __name__ == "__main__":
    main()
