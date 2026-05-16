"""
Step 5c: ROME causal trace — 完整 sweep，使用校准后的有效 corruption
  σ = 100 × embed_std ≈ 3.05 (绝对值)，在 subject ' Gemma' (S1) 上
  → corrupted p(' Google') ≈ 0.02

  对每个 (layer L, position p)：
    在 corrupted run 中把 layer L 输入的 residual 在 position p
    替换为 clean run 的同位置 residual
  → 测 p_restored(' Google') at last_pos

  IE = p_restored - p_corrupted
  argmax IE = critical fact-storage site

测两个 subject 选择：
  S1 = ' Gemma'        单 token
  S2 = ' Gemma 4'      3 tokens
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

SIGMA_ABS = 3.05    # ≈ 100 × embed_std
N_TRIALS = 5         # average over noise seeds for sweep (1 trial = 41s, 5 = ~205s × 2 subjects)
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
    print(f"[seq] T={seq_len}  assistant_start={assistant_start}", flush=True)

    SUBJECTS = {
        "S1_Gemma":  [13],
        "S2_Gemma4": [13, 14, 15],
    }

    # ---- clean run, capture residuals ----
    clean_residual = [None] * n_layers
    def make_clean_pre(L):
        def pre(module, inputs):
            clean_residual[L] = inputs[0][0].detach().clone()
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
        p_g_clean = clean_probs[GOOGLE_ID].item()
        p_a_clean = clean_probs[APPLE_ID].item()
        print(f"[clean] p(' Google')={p_g_clean:.4f}  p(' Apple')={p_a_clean:.4f}",
              flush=True)
    finally:
        for h in handles:
            h.remove()

    # ---- corrupt+restore run helper ----
    def run(noise_pos, sigma=SIGMA_ABS, restore_layer=None, restore_position=None,
            seed_offset=0):
        ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + seed_offset)

        def embed_hook(module, inputs, output):
            x = output.clone()
            for p in noise_pos:
                noise = torch.randn(x.shape[-1], generator=gen, device=DEVICE,
                                     dtype=x.dtype) * sigma
                x[0, p, :] = x[0, p, :] + noise
            return x

        h_embed = embed_tokens.register_forward_hook(embed_hook)
        h_restore = None
        if restore_layer is not None:
            cv = clean_residual[restore_layer][restore_position]
            def restore_pre(module, inputs):
                x = inputs[0].clone()
                x[0, restore_position, :] = cv
                return (x,) + inputs[1:]
            h_restore = layers[restore_layer].register_forward_pre_hook(restore_pre)

        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
            return probs[GOOGLE_ID].item(), probs[APPLE_ID].item()
        finally:
            h_embed.remove()
            if h_restore is not None: h_restore.remove()

    all_results = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sub_name, sub_pos in SUBJECTS.items():
        print(f"\n{'='*70}\n  Subject: {sub_name}  positions={sub_pos}\n{'='*70}",
              flush=True)

        # ---- corrupted baseline (avg N_TRIALS) ----
        pg_list = []
        for trial in range(N_TRIALS):
            pg, pa = run(sub_pos, seed_offset=trial)
            pg_list.append(pg)
        p_g_corrupt = sum(pg_list) / len(pg_list)
        print(f"[corrupt] avg p(' Google') over {N_TRIALS} trials = {p_g_corrupt:.4f}  "
              f"(drop {p_g_clean - p_g_corrupt:.4f})", flush=True)

        # ---- sweep ----
        # 35 layers × 19 positions × N_TRIALS = ~3325 forwards × ~62ms = ~206s
        sweep_pg = torch.zeros(n_layers, seq_len)
        t0 = time.time()
        for L in range(n_layers):
            for p in range(seq_len):
                pgs = []
                for trial in range(N_TRIALS):
                    pg, _ = run(sub_pos, restore_layer=L, restore_position=p,
                                  seed_offset=trial)
                    pgs.append(pg)
                sweep_pg[L, p] = sum(pgs) / len(pgs)
            print(f"  L{L:>2} done  ({time.time()-t0:.1f}s elapsed)", flush=True)
        sweep_IE = sweep_pg - p_g_corrupt

        # ---- analyze ----
        flat = sweep_IE.flatten()
        topv, topi = flat.topk(20)
        crit = []
        for v, idx in zip(topv.tolist(), topi.tolist()):
            L_, p_ = idx // seq_len, idx % seq_len
            tok_str = repr(tok.decode([seq[p_]], skip_special_tokens=False))
            gen_pos = p_ - assistant_start
            gp = f"gen{gen_pos}" if gen_pos >= 0 else f"prompt{p_}"
            crit.append({"L": L_, "pos": p_, "gen_pos": gen_pos, "IE": v,
                         "p_g_restored": sweep_pg[L_, p_].item(), "tok": tok_str})
            print(f"    L{L_:>2} pos{p_:>2} ({gp}) {tok_str:<22}  "
                  f"IE={v:+.4f}  p(' Google')={sweep_pg[L_, p_].item():.4f}",
                  flush=True)

        all_results[sub_name] = {
            "subject_positions": sub_pos,
            "p_g_corrupt": p_g_corrupt,
            "sweep_pg": sweep_pg, "sweep_IE": sweep_IE,
            "crit_sites": crit,
        }

        # ---- text heatmap ----
        with open(OUT_DIR / f"step5c_heatmap_{sub_name}.txt", "w", encoding="utf-8") as f:
            f.write(f"## ROME causal trace heatmap  subject={sub_name} pos={sub_pos}\n")
            f.write(f"## p_clean={p_g_clean:.4f}  p_corrupt={p_g_corrupt:.4f}  σ={SIGMA_ABS}\n")
            f.write(f"## IE = p_restored(' Google') - p_corrupt   higher = more critical\n\n")
            f.write(f"## scale: ' '<0.05  '·'<0.10  '+'<0.30  '*'<0.60  '#'≥0.60\n\n")

            # column header tokens
            f.write(f"  pos:    " + " ".join(f"{p:>2}" for p in range(seq_len)) + "\n")
            f.write(f"  token:  " + " ".join(
                f"{repr(tok.decode([seq[p]], skip_special_tokens=False))[:2]:>2}"
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
                # add max IE this layer
                lmax = sweep_IE[L].max().item()
                lmax_p = sweep_IE[L].argmax().item()
                row += f"   max={lmax:+.3f}@p{lmax_p}"
                f.write(row + "\n")

            # detailed table
            f.write(f"\n\n## detailed values: rows=layers, cols=positions\n")
            f.write(f"  pos     " + " ".join(f"{p:>6}" for p in range(seq_len)) + "\n")
            for L in range(n_layers):
                vals = " ".join(f"{sweep_IE[L, p].item():>6.3f}"
                                 for p in range(seq_len))
                f.write(f"  L{L:>2}    {vals}\n")

            # token reference
            f.write(f"\n\n## token reference\n")
            for p in range(seq_len):
                tok_str = repr(tok.decode([seq[p]], skip_special_tokens=False))
                gen_pos = p - assistant_start
                gp = f"gen{gen_pos}" if gen_pos >= 0 else f"prompt{p}"
                f.write(f"  pos {p:>2}  ({gp})  id={seq[p]:>7}  {tok_str}\n")

        print(f"\n[save] step5c_heatmap_{sub_name}.txt", flush=True)

    # ---- save raw ----
    torch.save({
        "config": {
            "prompt": PROMPT_TEXT, "seq": seq, "assistant_start": assistant_start,
            "last_pos": last_pos, "subjects": SUBJECTS,
            "sigma_abs": SIGMA_ABS, "n_trials": N_TRIALS,
            "google_id": GOOGLE_ID, "apple_id": APPLE_ID,
        },
        "p_g_clean": p_g_clean,
        "p_a_clean": p_a_clean,
        "all_results": all_results,
    }, OUT_DIR / "step5c_full_sweep_raw.pt")
    print(f"[save] step5c_full_sweep_raw.pt", flush=True)


if __name__ == "__main__":
    main()
