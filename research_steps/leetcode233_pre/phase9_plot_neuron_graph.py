"""
Phase 9: plot neuron relationship structure graph (matplotlib, ASCII only).

Y-axis: layer (early -> late, top to bottom)
Each neuron node labeled with:
  - id (L##)
  - IN: top-3 ASCII tokens that W_gate column projects onto (lm_head proj)
        = "what this gate reads as detection signal"
  - OUT: top-3 ASCII tokens that W_down column projects onto
        = "what this neuron pushes (output direction)"
Color: source tier
  - red    = K=30 negative-diff (suppressed-by-base)
  - blue   = K=30 positive-diff
  - orange = Phase 8b router candidate (new, not in K=30)
  - green  = Phase 4c executor candidate (Bug A path)
  - purple = Phase 4b' executor (Bug B path)

NO Chinese (matplotlib default font 'DejaVu Sans' for ASCII).
Output: J:/amr/amr_wtf/outputs/gemma_code_GB01/phase9_neuron_graph.png
"""
import os, sys, re
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"   # ASCII safe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase9_neuron_graph.png"

DEVICE = "cuda:0"

# Hand-picked neurons across all phases
# (L, I, tier, diff_or_contrib)
NEURONS = [
    # === Phase 8b router candidates (NEW, not in K=30, mid-layer) ===
    (15, 706,   "router",       +0.82),
    (18, 10758, "router",       +0.92),
    (18, 11537, "router",       -0.89),
    (23, 9585,  "router",       +1.09),
    (23, 11128, "router",       +0.93),
    (24, 10935, "router",       -1.00),

    # === Phase 2 K=30 clamp set (top 15 by |diff|) ===
    (15, 11943, "K30_neg",      -0.99),
    (16, 8035,  "K30_neg",      -1.49),
    (17, 7444,  "K30_neg",      -1.75),   # also Phase 8b top-1
    (23, 2564,  "K30_neg",      -0.99),
    (25, 7969,  "K30_neg",      -0.96),
    (25, 10558, "K30_pos",      +1.43),
    (25, 8980,  "K30_pos",      +1.34),
    (26, 10136, "K30_neg",      -4.20),   # top-1 negative diff
    (26, 10216, "K30_neg",      -1.15),
    (26, 12271, "K30_pos",      +1.07),
    (27, 4115,  "K30_neg",      -1.62),
    (27, 3892,  "K30_neg",      -1.13),
    (27, 10990, "K30_pos",      +0.99),
    (27, 8010,  "K30_pos",      +0.98),
    (33, 9054,  "K30_pos",      +1.27),
    (28, 8466,  "K30_neg",      -1.04),
    (34, 7490,  "K30_pos",      +0.95),

    # === Phase 4c Bug A executors (cot 'D>1 -> P' push) ===
    (33, 11786, "BugA_exec",    +1.41),   # cleanest
    (32, 5342,  "BugA_exec",    +0.85),
    (26, 1768,  "BugA_exec",    +0.78),

    # === Phase 4b' Bug B executors (code 'n /= 10' push) ===
    (34, 9243,  "BugB_exec",    +2.48),
    (34, 3988,  "BugB_exec",    +0.70),
]


TIER_COLOR = {
    "router":    "#FF8C00",  # orange
    "K30_neg":   "#D32F2F",  # red
    "K30_pos":   "#1976D2",  # blue
    "BugA_exec": "#388E3C",  # green
    "BugB_exec": "#7B1FA2",  # purple
}

TIER_LABEL = {
    "router":    "router (mid-layer, fires at digit-topic)",
    "K30_neg":   "K=30 clamp (base-side, suppressed by clamp)",
    "K30_pos":   "K=30 clamp (opus-side, boosted by clamp)",
    "BugA_exec": "Bug A executor (push '0' over '$P$')",
    "BugB_exec": "Bug B executor (push 'n' over 'P')",
}


def is_ascii_clean(s):
    """Token is mostly ASCII printable (no CJK, no foreign script)."""
    if not s.strip():
        return False
    return all(c.isascii() and (c.isalnum() or c in " /_.,-+()[]<>{}*#$%@!?:;'\"\\") for c in s)


def top_ascii_tokens(proj, tok, k=3, descending=True):
    """Return top-k ASCII-clean tokens from a projection tensor."""
    order = proj.argsort(descending=descending)
    out = []
    for ti in order.tolist():
        s = tok.decode([ti], skip_special_tokens=False)
        s_clean = s.strip()
        if is_ascii_clean(s):
            # remove leading space for display
            disp = s.lstrip()
            if len(disp) > 0 and len(disp) <= 12:
                out.append(disp)
                if len(out) >= k:
                    break
    return out


def main():
    print("[load] tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    lm_head_W = model.lm_head.weight.to(torch.float32).cpu()  # [vocab, hidden]

    print(f"[compute] {len(NEURONS)} neurons IN/OUT labels...")
    node_info = []  # (L, I, tier, diff, in_labels, out_labels)
    for L, I, tier, diff in NEURONS:
        W_gate_col = layers[L].mlp.gate_proj.weight.data[I, :].to(torch.float32).cpu()
        W_down_col = layers[L].mlp.down_proj.weight.data[:, I].to(torch.float32).cpu()
        # IN = W_gate column @ lm_head.T (what gate reads)
        gate_proj_vocab = W_gate_col @ lm_head_W.T
        # OUT = W_down column @ lm_head.T (what neuron pushes)
        down_proj_vocab = W_down_col @ lm_head_W.T

        in_top = top_ascii_tokens(gate_proj_vocab, tok, k=3, descending=True)
        out_push = top_ascii_tokens(down_proj_vocab, tok, k=3, descending=True)
        out_suppress = top_ascii_tokens(down_proj_vocab, tok, k=3, descending=False)
        node_info.append((L, I, tier, diff, in_top, out_push, out_suppress))
        print(f"  L{L:02d}#{I:<6d} [{tier}]  IN: {in_top}  OUT+: {out_push}  OUT-: {out_suppress}")

    # === Plot ===
    print("\n[plot]...")
    fig, ax = plt.subplots(figsize=(20, 18))

    # Group by layer
    by_layer = {}
    for n in node_info:
        L = n[0]
        by_layer.setdefault(L, []).append(n)

    # Position nodes: y = -L (so early layers top), x within layer = spread
    all_y = sorted(by_layer.keys())
    node_positions = {}
    for L in all_y:
        nodes = by_layer[L]
        n = len(nodes)
        # spread x in [-n/2, n/2] * spacing
        for i, info in enumerate(nodes):
            x = (i - (n - 1) / 2) * 5.0
            y = -L
            node_positions[(info[0], info[1])] = (x, y, info)

    # Draw layer guidelines
    L_min, L_max = min(all_y), max(all_y)
    for L in range(L_min, L_max + 1):
        ax.axhline(y=-L, color='lightgray', linestyle=':', linewidth=0.5, zorder=0)
        ax.text(-25, -L, f"L{L:02d}", fontsize=10, va='center',
                color='#555', weight='bold')

    # Draw nodes
    for (L, I), (x, y, info) in node_positions.items():
        L, I, tier, diff, in_top, out_push, out_suppress = info
        color = TIER_COLOR[tier]

        # box
        box = FancyBboxPatch(
            (x - 2.2, y - 0.45), 4.4, 0.9,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.6,
            zorder=2
        )
        ax.add_patch(box)

        # title (id + diff)
        ax.text(x, y + 0.25, f"L{L:02d}#{I}",
                ha='center', va='center', fontsize=9, weight='bold',
                color=color, zorder=4)
        ax.text(x, y + 0.06, f"d={diff:+.2f}",
                ha='center', va='center', fontsize=7.5, color='#333', zorder=4)

        # IN / OUT labels
        in_str = ", ".join(in_top[:2]) if in_top else "-"
        out_str = ", ".join(out_push[:2]) if out_push else "-"
        ax.text(x, y - 0.13, f"IN: {in_str}",
                ha='center', va='center', fontsize=6.5, style='italic',
                color='#444', zorder=4)
        ax.text(x, y - 0.30, f"OUT+: {out_str}",
                ha='center', va='center', fontsize=6.5,
                color='#222', zorder=4)

    # Legend
    legend_y = -L_max - 2
    for i, (tier, label) in enumerate(TIER_LABEL.items()):
        col = TIER_COLOR[tier]
        ax.add_patch(FancyBboxPatch(
            (-30 + i * 22, legend_y - 0.3), 1.5, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=col, alpha=0.18, edgecolor=col, linewidth=1.4
        ))
        ax.text(-28 + i * 22, legend_y, label,
                fontsize=8, va='center', color=col, weight='bold')

    # Title
    ax.text(0, -L_min + 2,
            "leetcode 233 neuron structure  |  IN = W_gate@lm_head top-2 ASCII  |  "
            "OUT+ = W_down@lm_head top-2 push direction (ASCII)",
            ha='center', fontsize=12, weight='bold')
    ax.text(0, -L_min + 1.2,
            "(early layers TOP, late layers BOTTOM; d = phase2 diff or phase4c contrib)",
            ha='center', fontsize=9, color='#666', style='italic')

    # x/y limits
    all_x = [p[0] for p in node_positions.values()]
    ax.set_xlim(min(all_x) - 8, max(all_x) + 8)
    ax.set_ylim(legend_y - 2, -L_min + 3)
    ax.set_aspect('auto')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
