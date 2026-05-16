"""
Step 21: down_col 几何邻居 + 下游听众分析

Part A — 空间 2: 找每个 target 神经元的 down_col 几何邻居
  对每个 target 算它的 down_col 跟所有其他神经元 down_col 的 cos sim
  找 top-K 邻居 + 跟 inventory 已知神经元对照

Part B — 空间 3: 找 L27#6644 (d3_depth) 的下游"听众"
  对 L28-L34 各层每个神经元，算 gate_proj[i, :] 和 up_proj[i, :]
  跟 target down_col 的 cos sim，找 gate*up 都对齐的下游神经元
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
INV_PATH = Path("J:/amr/amr_wtf/chat/neurons.json")
OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")

TARGETS = [
    ("L28#2406", 28, 2406, "fact: Google"),
    ("L27#6644", 27, 6644, "lit/d3_depth_pos"),
    ("L27#10024",27, 10024,"lit/d1_detail_neg"),
    ("L27#5590", 27, 5590, "lit/d3_depth_neg"),
    ("L32#9383", 32, 9383, "lit/d2_structure_neg"),
    ("L32#8474", 32, 8474, "lit/d1_detail_pos"),
    ("L34#8522", 34, 8522, "lit/d2_marker_pos"),
    ("L34#6608", 34, 6608, "thought/SOP_template_neg"),
]

TOP_K = 25
DOWNSTREAM_TARGET = ("L27#6644", 27, 6644, "lit/d3_depth_pos")
DOWNSTREAM_TOP = 15


def main():
    # load inventory
    inv = json.loads(INV_PATH.read_text(encoding="utf-8"))
    inv_map = {(n["layer"], n["index"]): n["label"] for n in inv["known_neurons"]}
    print(f"[inv] loaded {len(inv_map)} known neurons", flush=True)

    print(f"[load] {MODEL_PATH}", flush=True)
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
    n_layers = len(layers)

    # ---- 预先 normalize 所有层的 W_down 列 (方便后续 cos sim) ----
    print(f"[setup] normalizing W_down across {n_layers} layers ...", flush=True)
    layer_dims = []
    layer_W_down_n = []  # 存归一化后的 W_down (在 GPU)
    for L in range(n_layers):
        W = layers[L].mlp.down_proj.weight.float()  # [hidden=1536, intermediate]
        W_n = F.normalize(W, dim=0)                 # 列方向归一
        layer_W_down_n.append(W_n)
        layer_dims.append(W.shape[1])
    total_neurons = sum(layer_dims)
    print(f"  total neurons: {total_neurons}", flush=True)

    # ---- Part A: 几何邻居 ----
    print(f"\n{'='*100}\n  Part A — down_col 几何邻居 (top-{TOP_K})\n{'='*100}",
          flush=True)

    partA_results = {}
    for tid, L_t, i_t, label in TARGETS:
        target_v = layers[L_t].mlp.down_proj.weight[:, i_t].float()
        target_v_n = F.normalize(target_v.unsqueeze(0), dim=1).squeeze(0)  # [1536]

        # 跟每层 W_down 算 cos sim
        all_sims = []   # (L, i, sim)
        for L in range(n_layers):
            sims = target_v_n @ layer_W_down_n[L]  # [intermediate_L]
            sims_cpu = sims.cpu()
            for i in range(layer_dims[L]):
                if (L, i) == (L_t, i_t): continue
                all_sims.append((L, i, sims_cpu[i].item()))
        all_sims.sort(key=lambda x: -x[2])

        print(f"\n--- {tid}  ({label}) ---", flush=True)
        partA_results[tid] = []
        for rank, (L, i, s) in enumerate(all_sims[:TOP_K]):
            nname = f"L{L}#{i}"
            inv_label = inv_map.get((L, i), "")
            tag = f"  ★ {inv_label}" if inv_label else ""
            print(f"  {rank+1:>2}. {nname:<11} cos={s:>+7.3f}{tag}", flush=True)
            partA_results[tid].append({
                "rank": rank+1, "L": L, "i": i, "cos": s,
                "inv_label": inv_label,
            })

    # ---- Part B: 下游听众 (L27#6644 → L28-L34 gate*up) ----
    tid_b, L_b, i_b, label_b = DOWNSTREAM_TARGET
    print(f"\n{'='*100}\n  Part B — {tid_b} ({label_b}) 下游听众\n  ↓\n  gate_proj[L', i] · down_col[27, 6644]  和\n  up_proj[L', i]   · down_col[27, 6644]  都对齐的 i\n{'='*100}",
          flush=True)
    target_v = layers[L_b].mlp.down_proj.weight[:, i_b].float()
    target_v_n = F.normalize(target_v.unsqueeze(0), dim=1).squeeze(0)  # [1536]

    partB_results = {}
    for L_down in range(L_b + 1, n_layers):
        gate_W = layers[L_down].mlp.gate_proj.weight.float()   # [intermediate, hidden]
        up_W = layers[L_down].mlp.up_proj.weight.float()
        # row-wise normalize
        gate_W_n = F.normalize(gate_W, dim=1)
        up_W_n = F.normalize(up_W, dim=1)

        gate_sims = (gate_W_n @ target_v_n)   # [intermediate]
        up_sims = (up_W_n @ target_v_n)
        combined = gate_sims * up_sims          # 两者同号且都大才高

        top_idx = combined.argsort(descending=True)[:DOWNSTREAM_TOP].cpu().tolist()
        print(f"\n--- L{L_down} 听众 (top-{DOWNSTREAM_TOP} by gate·up) ---", flush=True)
        partB_results[L_down] = []
        for idx in top_idx:
            g = gate_sims[idx].item()
            u = up_sims[idx].item()
            c = combined[idx].item()
            inv_label = inv_map.get((L_down, idx), "")
            tag = f"  ★ {inv_label}" if inv_label else ""
            print(f"  L{L_down}#{idx:<5}  g={g:+.3f}  u={u:+.3f}  g*u={c:+.4f}{tag}",
                  flush=True)
            partB_results[L_down].append({
                "L": L_down, "i": idx, "gate_sim": g, "up_sim": u,
                "combined": c, "inv_label": inv_label,
            })

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step21_geometric_neighbors.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 21: 几何邻居 + 下游听众 ================\n\n")

        f.write(f"\n=== Part A: down_col 几何邻居 (top-{TOP_K}) ===\n")
        for tid, L, i, label in TARGETS:
            f.write(f"\n--- {tid}  ({label}) ---\n")
            for r in partA_results[tid]:
                tag = f"  ★ {r['inv_label']}" if r['inv_label'] else ""
                f.write(f"  {r['rank']:>2}. L{r['L']}#{r['i']:<5}  "
                        f"cos={r['cos']:>+7.3f}{tag}\n")

        f.write(f"\n\n=== Part B: {tid_b} 下游听众 ===\n")
        for L_down in range(L_b + 1, n_layers):
            f.write(f"\n--- L{L_down} (top-{DOWNSTREAM_TOP}) ---\n")
            for r in partB_results[L_down]:
                tag = f"  ★ {r['inv_label']}" if r['inv_label'] else ""
                f.write(f"  L{L_down}#{r['i']:<5}  "
                        f"g={r['gate_sim']:+.3f}  u={r['up_sim']:+.3f}  "
                        f"g*u={r['combined']:+.4f}{tag}\n")

    torch.save({
        "targets": TARGETS,
        "partA": partA_results,
        "partB": partB_results,
        "downstream_target": DOWNSTREAM_TARGET,
    }, OUT_DIR / "step21_geometric_neighbors_raw.pt")
    print(f"\n[save] step21_geometric_neighbors.txt", flush=True)


if __name__ == "__main__":
    main()
