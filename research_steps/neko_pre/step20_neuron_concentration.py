"""
量化 W_down 列在 vocab 上的"集中度":
  fact 神经元应当极集中 (top-15 占大部分 magnitude)
  风格神经元应当散布 (要 top-100 / top-1000 才占同样比例)
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"

NEURONS = [
    ("L28#2406", 28, 2406, "fact: Google 路由器"),
    ("L27#6644", 27, 6644, "lit/d3_depth_pos"),
    ("L27#10024",27, 10024,"lit/d1_detail_neg"),
    ("L27#5590", 27, 5590, "lit/d3_depth_neg"),
    ("L32#9383", 32, 9383, "lit/d2_structure_neg"),
    ("L32#8474", 32, 8474, "lit/d1_detail_pos"),
    ("L34#8522", 34, 8522, "lit/d2_marker_pos"),
    ("L34#6608", 34, 6608, "thought/SOP_template_neg"),
    ("L27#2890", 27, 2890, "general/long_struct_pos"),
]


def main():
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
    layers = model.model.language_model.layers
    W_un = model.lm_head.weight.float().cpu()   # [vocab=262144, hidden]
    print(f"vocab size: {W_un.shape[0]}", flush=True)

    Ks = [1, 5, 15, 50, 100, 500, 1000, 5000]
    print(f"\n{'name':<11} {'L#idx':<10} {'label':<32} {'norm':>6} ", end="")
    for K in Ks:
        print(f" Top{K:>4}%", end="")
    print(f"  {'eff_rank':>8}  {'entropy':>8}", flush=True)
    print(f"{'-'*11} {'-'*10} {'-'*32} {'-'*6} ", end="")
    for K in Ks:
        print(f" {'-'*7}", end="")
    print(f"  {'-'*8}  {'-'*8}", flush=True)

    rows = []
    for nid, layer, idx, label in NEURONS:
        W_down = layers[layer].mlp.down_proj.weight.data.float().cpu()
        v = W_down[:, idx]                       # [hidden]
        norm_v = v.norm().item()

        proj = W_un @ v                          # [vocab]
        abs_proj = proj.abs()
        sorted_proj = abs_proj.sort(descending=True).values
        total = sorted_proj.sum().item()

        # 各 K 累积占比
        cum_pcts = []
        for K in Ks:
            pct = sorted_proj[:K].sum().item() / total * 100
            cum_pcts.append(pct)

        # effective rank: 1 / Σ(p²) 其中 p_i = abs_proj_i / total
        p = abs_proj / total
        eff_rank = 1.0 / (p * p).sum().item()

        # entropy
        p_safe = p.clamp(min=1e-12)
        entropy = -(p_safe * p_safe.log()).sum().item()

        print(f"{nid:<11} L{layer}#{idx:<7} {label[:32]:<32} {norm_v:>6.3f} ", end="")
        for pct in cum_pcts:
            print(f" {pct:>6.2f}%", end="")
        print(f"  {eff_rank:>8.0f}  {entropy:>8.3f}", flush=True)
        rows.append({"id": nid, "label": label, "norm": norm_v,
                     "cum_pcts": dict(zip(Ks, cum_pcts)),
                     "eff_rank": eff_rank, "entropy": entropy})

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step20_concentration.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 20: W_down 列 vocab 投影集中度 ================\n\n")
        f.write(f"vocab size = {W_un.shape[0]}\n\n")
        f.write(f"{'name':<11} {'label':<34} {'norm':>6} ")
        for K in Ks: f.write(f" Top{K:>4}%")
        f.write(f"  {'eff_rank':>9}  {'entropy':>8}\n")
        for r in rows:
            f.write(f"{r['id']:<11} {r['label'][:34]:<34} {r['norm']:>6.3f} ")
            for K in Ks:
                f.write(f" {r['cum_pcts'][K]:>6.2f}%")
            f.write(f"  {r['eff_rank']:>9.0f}  {r['entropy']:>8.3f}\n")
        f.write(f"\neff_rank = 1/Σp²  (p_i = |proj_i| / Σ|proj|)\n")
        f.write(f"entropy = -Σ p log p  (越大 = 越散布)\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
