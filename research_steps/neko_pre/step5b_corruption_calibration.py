"""
Step 5b: corruption 强度校准
  step 5 的 σ=3×embed_std 在 Gemma 4 上完全失效（embedding 后乘 sqrt(1536)
  + RMSNorm 抹掉了噪声）。需要找出真正能破坏 fact-recall 的 corruption。

测试 6 种 corruption 策略，目标：把 p(' Google') 从 1.0 拉低到 < 0.5：

  A. σ × (3, 10, 30, 100, 300, 1000) of embed_std        # noise sweep
  B. token replacement: ' Gemma' → ' Llama' / ' GPT' / ' Apple' / 完全乱码 token
  C. zero embed                                           # 直接清零
  D. noise on layer-0 hidden (after embed scale)         # 改注入点
  E. multi-token subject (S2 / S3) + 大 σ
  F. 对所有 prompt+subject 整段 embed 加噪声

找出最有效的策略，输出推荐设置给 step 5c (full sweep)。

Output: identity_swap/step5b_calibration.txt
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
N_TRIALS = 5


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

    embed_std = embed_tokens.weight.detach().float().std().item()
    print(f"[noise] embed_std={embed_std:.4f}", flush=True)

    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq_base = prefix_ids + R_PREFIX_IDS
    assistant_start = len(prefix_ids)
    seq_len = len(seq_base)
    last_pos = seq_len - 1
    print(f"[seq] T={seq_len}  decode={tok.decode(seq_base, skip_special_tokens=False)!r}",
          flush=True)

    def run(seq_ids, embed_noise=None, embed_zero_pos=None,
            l0_hidden_noise=None):
        """
        Run forward with optional corruption strategies.
          embed_noise:    (positions, sigma)    add Gaussian to embed_tokens output
          embed_zero_pos: positions             zero out embed at these positions
          l0_hidden_noise: (positions, sigma)   add Gaussian to layer 0 hidden input
        """
        ids_t = torch.tensor([seq_ids], dtype=torch.long, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(0)

        def embed_hook(module, inputs, output):
            x = output.clone()
            if embed_noise is not None:
                pos_list, sigma = embed_noise
                for p in pos_list:
                    noise = torch.randn(x.shape[-1], generator=gen, device=DEVICE,
                                         dtype=x.dtype) * sigma
                    x[0, p, :] = x[0, p, :] + noise
            if embed_zero_pos is not None:
                for p in embed_zero_pos:
                    x[0, p, :] = 0
            return x

        def l0_pre(module, inputs):
            x = inputs[0].clone()
            if l0_hidden_noise is not None:
                pos_list, sigma = l0_hidden_noise
                for p in pos_list:
                    noise = torch.randn(x.shape[-1], generator=gen, device=DEVICE,
                                         dtype=x.dtype) * sigma
                    x[0, p, :] = x[0, p, :] + noise
            return (x,) + inputs[1:]

        h_embed = embed_tokens.register_forward_hook(embed_hook)
        h_l0 = layers[0].register_forward_pre_hook(l0_pre) if l0_hidden_noise else None

        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            probs = F.softmax(out.logits[0, last_pos].float(), dim=-1)
            return probs[GOOGLE_ID].item(), probs[APPLE_ID].item()
        finally:
            h_embed.remove()
            if h_l0 is not None:
                h_l0.remove()

    def trial_avg(fn):
        results = [fn() for _ in range(N_TRIALS)]
        pg = sum(r[0] for r in results) / len(results)
        pa = sum(r[1] for r in results) / len(results)
        return pg, pa

    # subject positions
    SUB_POS_S1 = [13]                # ' Gemma'
    SUB_POS_S2 = [13, 14, 15]        # ' Gemma 4'
    SUB_POS_S3 = [12, 13, 14, 15]    # '我是 Gemma 4'

    lines = ["================ Step 5b: corruption 校准 ================\n\n"]
    lines.append(f"embed_std = {embed_std:.4f}\n")
    lines.append(f"target: corrupt subject to drop p(' Google') from 1.0 → < 0.5\n\n")

    print(f"\n{'='*80}\n  Strategy A: σ sweep on embed (subject = ' Gemma' alone)\n{'='*80}")
    lines.append(f"\n--- A: σ sweep on embed @ S1=' Gemma' ---\n")
    for mult in [3, 10, 30, 100, 300, 1000]:
        sigma = mult * embed_std
        pg, pa = trial_avg(lambda: run(seq_base, embed_noise=(SUB_POS_S1, sigma)))
        ln = f"  σ={mult}×std ({sigma:.3f})  p(Google)={pg:.4f}  p(Apple)={pa:.4f}"
        print(ln, flush=True); lines.append(ln + "\n")

    print(f"\n{'='*80}\n  Strategy A': σ sweep on embed (subject = ' Gemma 4')\n{'='*80}")
    lines.append(f"\n--- A': σ sweep @ S2=' Gemma 4' ---\n")
    for mult in [3, 10, 30, 100, 300, 1000]:
        sigma = mult * embed_std
        pg, pa = trial_avg(lambda: run(seq_base, embed_noise=(SUB_POS_S2, sigma)))
        ln = f"  σ={mult}×std ({sigma:.3f})  p(Google)={pg:.4f}  p(Apple)={pa:.4f}"
        print(ln, flush=True); lines.append(ln + "\n")

    print(f"\n{'='*80}\n  Strategy B: token replacement (single token swap)\n{'='*80}")
    lines.append(f"\n--- B: token replacement (' Gemma' → ?) ---\n")
    REPL = {
        " Llama":  tok.encode(" Llama",  add_special_tokens=False),
        " GPT":    tok.encode(" GPT",    add_special_tokens=False),
        " Claude": tok.encode(" Claude", add_special_tokens=False),
        " Apple":  [9947],
        " Banana": tok.encode(" Banana", add_special_tokens=False),
        " 苹果":    [68004],   # need w/o leading space — note pos 13 has leading-space ' Gemma'
        " random": [123456],   # random vocab id
    }
    for name, ids in REPL.items():
        if len(ids) != 1:
            ln = f"  {name!r:<14}  multi-token, skip ({ids})"
            print(ln, flush=True); lines.append(ln + "\n"); continue
        seq_repl = list(seq_base)
        seq_repl[13] = ids[0]      # replace ' Gemma' (abs pos 13)
        pg, pa = trial_avg(lambda s=seq_repl: run(s))
        ln = f"  ' Gemma' → {name!r:<10} (id {ids[0]:>6})  p(Google)={pg:.4f}  p(Apple)={pa:.4f}"
        print(ln, flush=True); lines.append(ln + "\n")

    print(f"\n{'='*80}\n  Strategy C: zero embed at subject positions\n{'='*80}")
    lines.append(f"\n--- C: zero embed at subject ---\n")
    for sub_name, sub_pos in [("S1", SUB_POS_S1), ("S2", SUB_POS_S2), ("S3", SUB_POS_S3)]:
        pg, pa = trial_avg(lambda: run(seq_base, embed_zero_pos=sub_pos))
        ln = f"  {sub_name} pos={sub_pos}  p(Google)={pg:.4f}  p(Apple)={pa:.4f}"
        print(ln, flush=True); lines.append(ln + "\n")

    print(f"\n{'='*80}\n  Strategy D: σ sweep on layer-0 hidden input (post-scale)\n{'='*80}")
    lines.append(f"\n--- D: σ sweep on layer-0 hidden input @ S1 ---\n")
    # layer 0 input has been scaled by sqrt(hidden) ≈ 39
    # Try absolute σ values
    for sigma in [0.1, 1.0, 5.0, 10.0, 30.0, 100.0]:
        pg, pa = trial_avg(lambda: run(seq_base, l0_hidden_noise=(SUB_POS_S1, sigma)))
        ln = f"  σ={sigma:>6.1f}  p(Google)={pg:.4f}  p(Apple)={pa:.4f}"
        print(ln, flush=True); lines.append(ln + "\n")

    print(f"\n{'='*80}\n  Strategy D': σ sweep on layer-0 hidden input (S2 = ' Gemma 4')\n{'='*80}")
    lines.append(f"\n--- D': σ sweep on layer-0 hidden input @ S2 ---\n")
    for sigma in [0.1, 1.0, 5.0, 10.0, 30.0, 100.0]:
        pg, pa = trial_avg(lambda: run(seq_base, l0_hidden_noise=(SUB_POS_S2, sigma)))
        ln = f"  σ={sigma:>6.1f}  p(Google)={pg:.4f}  p(Apple)={pa:.4f}"
        print(ln, flush=True); lines.append(ln + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step5b_calibration.txt").write_text("".join(lines), encoding="utf-8")
    print(f"\n[save] step5b_calibration.txt", flush=True)


if __name__ == "__main__":
    main()
