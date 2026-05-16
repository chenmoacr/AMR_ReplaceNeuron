"""
Step 5e: ROME causal trace — MLP-only restore (ROME 论文标准方法)
  step 5d 显示 residual restore 太宽松（任何 L,p 都恢复 fact）。
  ROME 论文用 "MLP output restore" 作为 sharper attribution：
    在 corrupt run 中，仅 restore layer L 的 MLP **输出**（不动 attn）
    在 position p 上替换 mlp output 为 clean run 的 mlp output
  → 揭示 MLP 在哪个 (L,p) 真正 critical to fact propagation

同时测 attn-only restore 作为对照：
  仅 restore layer L 的 self_attn output

最后做 IE 对比：MLP_IE vs Attn_IE per (L,p)

Output:
  step5e_mlp_attn_trace_raw.pt
  step5e_mlp_heatmap.txt
  step5e_attn_heatmap.txt
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
GEMMA_ID = 147224
GPT_ID   = 72375
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

    # find self_attn module name (Gemma 4 likely 'self_attn')
    print(f"[layer 0 modules] {list(dict(layers[0].named_children()).keys())}", flush=True)

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
    POS_GEMMA = assistant_start + 1
    seq_corrupt = list(seq_clean)
    seq_corrupt[POS_GEMMA] = GPT_ID

    # ---- 1. clean run: capture mlp.output and self_attn.output per layer ----
    clean_mlp_out  = [None] * n_layers
    clean_attn_out = [None] * n_layers

    def make_mlp_hook(L):
        def hook(module, inputs, output):
            # output: [1, T, hidden]
            clean_mlp_out[L] = output[0].detach().clone()
        return hook

    def make_attn_hook(L):
        def hook(module, inputs, output):
            # transformers attn returns tuple (hidden, attn_weights, past_kv) usually
            x = output[0] if isinstance(output, tuple) else output
            clean_attn_out[L] = x[0].detach().clone()
        return hook

    handles = []
    for L in range(n_layers):
        handles.append(layers[L].mlp.register_forward_hook(make_mlp_hook(L)))
        handles.append(layers[L].self_attn.register_forward_hook(make_attn_hook(L)))

    try:
        ids_t = torch.tensor([seq_clean], dtype=torch.long, device=DEVICE)
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        clean_probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
        p_g_clean = clean_probs[GOOGLE_ID].item()
    finally:
        for h in handles: h.remove()
    print(f"[clean]    p(' Google')={p_g_clean:.4f}", flush=True)
    print(f"  mlp_out  shape: {tuple(clean_mlp_out[0].shape)}", flush=True)
    print(f"  attn_out shape: {tuple(clean_attn_out[0].shape)}", flush=True)

    # ---- 2. corrupt baseline ----
    ids_c = torch.tensor([seq_corrupt], dtype=torch.long, device=DEVICE)
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model(input_ids=ids_c, use_cache=False)
    corr_probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
    p_g_corr = corr_probs[GOOGLE_ID].item()
    top5 = corr_probs.topk(5)
    print(f"[corrupt]  p(' Google')={p_g_corr:.4f}  top5: " +
          " / ".join(f"{repr(tok.decode([i.item()]))}={p.item():.3f}"
                      for p, i in zip(top5.values, top5.indices)), flush=True)

    # ---- 3. MLP-only restore sweep ----
    def run_mlp_restore(L, p):
        cv = clean_mlp_out[L][p]
        def hook(module, inputs, output):
            out = output.clone()
            out[0, p, :] = cv
            return out
        h = layers[L].mlp.register_forward_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_c, use_cache=False)
            return F.softmax(out.logits[0, last_pos].float(), dim=-1)[GOOGLE_ID].item()
        finally:
            h.remove()

    def run_attn_restore(L, p):
        cv = clean_attn_out[L][p]
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            x = x.clone()
            x[0, p, :] = cv
            if isinstance(output, tuple):
                return (x,) + output[1:]
            return x
        h = layers[L].self_attn.register_forward_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_c, use_cache=False)
            return F.softmax(out.logits[0, last_pos].float(), dim=-1)[GOOGLE_ID].item()
        finally:
            h.remove()

    print(f"\n[sweep MLP-only]  {n_layers} × {seq_len} forwards", flush=True)
    sweep_mlp = torch.zeros(n_layers, seq_len)
    t0 = time.time()
    for L in range(n_layers):
        for p in range(seq_len):
            sweep_mlp[L, p] = run_mlp_restore(L, p)
        if L % 5 == 4:
            print(f"  L{L:>2} done  ({time.time()-t0:.1f}s)", flush=True)
    IE_mlp = sweep_mlp - p_g_corr

    print(f"\n[sweep Attn-only] {n_layers} × {seq_len} forwards", flush=True)
    sweep_attn = torch.zeros(n_layers, seq_len)
    t0 = time.time()
    for L in range(n_layers):
        for p in range(seq_len):
            sweep_attn[L, p] = run_attn_restore(L, p)
        if L % 5 == 4:
            print(f"  L{L:>2} done  ({time.time()-t0:.1f}s)", flush=True)
    IE_attn = sweep_attn - p_g_corr

    # ---- analyze top sites ----
    def top_sites(IE, name, k=20):
        flat = IE.flatten()
        topv, topi = flat.topk(k)
        print(f"\n--- top {k} {name} restore sites ---")
        for v, idx in zip(topv.tolist(), topi.tolist()):
            L_, p_ = idx // seq_len, idx % seq_len
            tok_str = repr(tok.decode([seq_clean[p_]], skip_special_tokens=False))
            gen_pos = p_ - assistant_start
            gp = f"gen{gen_pos}" if gen_pos >= 0 else f"prompt{p_}"
            print(f"  L{L_:>2} pos{p_:>2} ({gp:>7}) {tok_str:<22}  IE={v:+.4f}",
                  flush=True)

    top_sites(IE_mlp, "MLP")
    top_sites(IE_attn, "Attn")

    # ---- write heatmaps ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    def write_heat(IE, name):
        path = OUT_DIR / f"step5e_{name}_heatmap.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"## ROME causal trace — {name}-only restore\n")
            f.write(f"## clean=Gemma  corrupt=GPT  p_clean={p_g_clean:.4f}  p_corr={p_g_corr:.4f}\n")
            f.write(f"## IE = p_restored(' Google') - p_corrupt   higher = critical\n")
            f.write(f"## scale: ' '<0.05  '·'<0.10  '+'<0.30  '*'<0.60  '#'≥0.60\n\n")
            f.write(f"  pos:    " + " ".join(f"{p:>2}" for p in range(seq_len)) + "\n")
            f.write(f"  token:  " + " ".join(
                f"{repr(tok.decode([seq_clean[p]], skip_special_tokens=False))[:2]:>2}"
                for p in range(seq_len)) + "\n")
            f.write(f"  -" * (seq_len * 3 + 6) + "\n")
            for L in range(n_layers):
                row = f"  L{L:>2}    "
                for p in range(seq_len):
                    v = IE[L, p].item()
                    if v < 0.05: ch = ' '
                    elif v < 0.10: ch = '·'
                    elif v < 0.30: ch = '+'
                    elif v < 0.60: ch = '*'
                    else: ch = '#'
                    row += f" {ch} "
                lmax = IE[L].max().item()
                lmax_p = IE[L].argmax().item()
                row += f"   max={lmax:+.3f}@p{lmax_p}"
                f.write(row + "\n")
            f.write(f"\n\n## detailed values\n")
            f.write(f"  pos     " + " ".join(f"{p:>6}" for p in range(seq_len)) + "\n")
            for L in range(n_layers):
                vals = " ".join(f"{IE[L, p].item():>6.3f}" for p in range(seq_len))
                f.write(f"  L{L:>2}    {vals}\n")
            f.write(f"\n\n## token reference\n")
            for p in range(seq_len):
                t = repr(tok.decode([seq_clean[p]], skip_special_tokens=False))
                gp = (p - assistant_start)
                gp_s = f"gen{gp}" if gp >= 0 else f"prompt{p}"
                f.write(f"  pos {p:>2}  ({gp_s})  id={seq_clean[p]:>7}  {t}\n")

    write_heat(IE_mlp, "mlp")
    write_heat(IE_attn, "attn")
    print(f"\n[save] step5e_mlp_heatmap.txt  step5e_attn_heatmap.txt", flush=True)

    torch.save({
        "config": {"prompt": PROMPT_TEXT, "seq_clean": seq_clean,
                    "seq_corrupt": seq_corrupt, "pos_gemma": POS_GEMMA},
        "p_g_clean": p_g_clean, "p_g_corr": p_g_corr,
        "sweep_mlp": sweep_mlp, "sweep_attn": sweep_attn,
        "IE_mlp": IE_mlp, "IE_attn": IE_attn,
    }, OUT_DIR / "step5e_mlp_attn_trace_raw.pt")
    print(f"[save] step5e_mlp_attn_trace_raw.pt", flush=True)


if __name__ == "__main__":
    main()
