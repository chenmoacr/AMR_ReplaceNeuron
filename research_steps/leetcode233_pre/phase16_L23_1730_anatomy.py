"""
Phase 16: L23#1730 完整内部解剖
- W_gate row → vocab projection: 它读什么 (gate detection signal)
- W_up row → vocab projection: 'up' branch 读什么
- W_down column → vocab projection: 它推/压什么 (output direction)
- phase4a 中 top 20 高 |act| 位置 + token context (它真正在哪些 context fire)
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
GEN_IDS_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_gen_ids.pt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase16_L23_1730.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
TARGET_L = 23
TARGET_I = 1730
TOP_VOCAB = 20
TOP_POS = 20
CONTEXT_BEFORE = 25
CONTEXT_AFTER = 8


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_hooks(layers, selections):
    by_layer = {}
    for it in selections:
        by_layer.setdefault(int(it["layer"]), []).append((int(it["index"]), float(it["gain"])))
    hooks = []
    for li, items in by_layer.items():
        layer_dev = layers[li].mlp.down_proj.weight.device
        idx = torch.tensor([n for n, _ in items], dtype=torch.long, device=layer_dev)
        shift_cpu = torch.tensor([s for _, s in items], dtype=torch.float32)
        cache = {}
        def make_fn(idx_local, shift_cpu_local, cache_local):
            def pre_hook(m, inputs):
                x = inputs[0]
                key = (x.dtype, x.device)
                shift_dev = cache_local.get(key)
                if shift_dev is None:
                    shift_dev = shift_cpu_local.to(dtype=x.dtype, device=x.device)
                    cache_local[key] = shift_dev
                x = x.clone()
                x[:, :, idx_local] = x[:, :, idx_local] + shift_dev
                return (x,) + inputs[1:]
            return pre_hook
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(
            make_fn(idx, shift_cpu, cache)))
    return hooks


def remove_hooks(hooks):
    for h in hooks: h.remove()


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
    lm_head = model.lm_head
    lm_head_W = lm_head.weight.to(torch.float32).cpu()

    L, I = TARGET_L, TARGET_I
    print(f"\n[target] L{L:02d}#{I}")
    W_gate_row = layers[L].mlp.gate_proj.weight.data[I, :].to(torch.float32).cpu()
    W_up_row = layers[L].mlp.up_proj.weight.data[I, :].to(torch.float32).cpu()
    W_down_col = layers[L].mlp.down_proj.weight.data[:, I].to(torch.float32).cpu()
    print(f"  ||W_gate row||={W_gate_row.norm().item():.3f}")
    print(f"  ||W_up   row||={W_up_row.norm().item():.3f}")
    print(f"  ||W_down col||={W_down_col.norm().item():.3f}")

    # Vocab projections
    proj_gate = W_gate_row @ lm_head_W.T   # [vocab]: tokens that the gate "reads positively"
    proj_up = W_up_row @ lm_head_W.T
    proj_down = W_down_col @ lm_head_W.T   # [vocab]: tokens that the output direction pushes
    print(f"  ||proj_gate||={proj_gate.norm().item():.3f}")
    print(f"  ||proj_up||={proj_up.norm().item():.3f}")
    print(f"  ||proj_down||={proj_down.norm().item():.3f}")

    def top_tokens(proj, k, descending=True):
        order = proj.argsort(descending=descending)
        out = []
        for ti in order.tolist():
            s = tok.decode([ti], skip_special_tokens=False)
            v = proj[ti].item()
            out.append((s, v, ti))
            if len(out) >= k:
                break
        return out

    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append(f"  Phase 16: L{L:02d}#{I} 内部解剖")
    out_lines.append("=" * 80)
    out_lines.append(f"||W_gate row||={W_gate_row.norm().item():.3f}  "
                     f"||W_up row||={W_up_row.norm().item():.3f}  "
                     f"||W_down col||={W_down_col.norm().item():.3f}")
    out_lines.append("")

    # ============= 激活条件 (gate detection) =============
    out_lines.append("=" * 80)
    out_lines.append("  激活条件 — W_gate row 在 vocab 上投影")
    out_lines.append("  (gate 读到 hidden 中含哪些 token 概念时, gate logit 高 → silu(gate) 大 → 激活)")
    out_lines.append("=" * 80)
    out_lines.append(f"\n[W_gate row top {TOP_VOCAB} PUSH (gate sees → fire):]")
    for s, v, ti in top_tokens(proj_gate, TOP_VOCAB, True):
        out_lines.append(f"  proj={v:+.4f}  id={ti:<7d}  {s!r}")
    out_lines.append(f"\n[W_gate row top {TOP_VOCAB} SUPPRESS (gate反向):]")
    for s, v, ti in top_tokens(proj_gate, TOP_VOCAB, False):
        out_lines.append(f"  proj={v:+.4f}  id={ti:<7d}  {s!r}")

    # ============= W_up branch =============
    out_lines.append("\n" + "=" * 80)
    out_lines.append("  W_up branch — second multiplicative branch in SwiGLU")
    out_lines.append("=" * 80)
    out_lines.append(f"\n[W_up row top {TOP_VOCAB} PUSH:]")
    for s, v, ti in top_tokens(proj_up, TOP_VOCAB, True):
        out_lines.append(f"  proj={v:+.4f}  id={ti:<7d}  {s!r}")
    out_lines.append(f"\n[W_up row top {TOP_VOCAB} SUPPRESS:]")
    for s, v, ti in top_tokens(proj_up, TOP_VOCAB, False):
        out_lines.append(f"  proj={v:+.4f}  id={ti:<7d}  {s!r}")

    # ============= 执行结果 (output direction) =============
    out_lines.append("\n" + "=" * 80)
    out_lines.append("  执行结果 — W_down column 在 vocab 上投影")
    out_lines.append("  (激活后, output 加到 residual stream, push/suppress 这些 token logit)")
    out_lines.append("=" * 80)
    out_lines.append(f"\n[W_down col top {TOP_VOCAB} PUSH (act>0 时升 logit):]")
    for s, v, ti in top_tokens(proj_down, TOP_VOCAB, True):
        out_lines.append(f"  proj={v:+.4f}  id={ti:<7d}  {s!r}")
    out_lines.append(f"\n[W_down col top {TOP_VOCAB} SUPPRESS (act>0 时降 logit):]")
    for s, v, ti in top_tokens(proj_down, TOP_VOCAB, False):
        out_lines.append(f"  proj={v:+.4f}  id={ti:<7d}  {s!r}")

    # ============= Forward 抓 phase4a 中 top 激活位置 =============
    print("\n[load cached body_ids]...")
    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    print(f"  {len(body_ids)} tokens")
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    T = full_ids.shape[0]

    p2 = torch.load(PHASE2_PT, weights_only=False)
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    captured_acts = []
    def hook(m, inputs):
        x = inputs[0][0]  # [T, 12288]
        captured_acts.append(x[:, I].to(torch.float32).cpu().clone())
    h = layers[L].mlp.down_proj.register_forward_pre_hook(hook)
    hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[forward] capture L{L}#{I} act per token...")
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    finally:
        remove_hooks(hooks)
        h.remove()
    acts = captured_acts[0]   # [T]
    print(f"  acts.shape={acts.shape}, range=[{acts.min().item():.2f}, {acts.max().item():.2f}]")

    # Only assistant region positions
    acts_assistant = acts.clone()
    acts_assistant[:prompt_len] = 0
    top_idx = acts_assistant.abs().topk(TOP_POS).indices.tolist()

    out_lines.append("\n" + "=" * 80)
    out_lines.append(f"  实际触发条件 — phase4a 中 L{L}#{I} top {TOP_POS} 高 |act| 位置 + 上下文")
    out_lines.append("=" * 80)
    out_lines.append("(K=30 clamp on, 在 assistant region 内, 排序 |act|)")
    out_lines.append(f"\n  global act stats: min={acts.min().item():.2f}  max={acts.max().item():.2f}  "
                     f"mean={acts.mean().item():.3f}  std={acts.std().item():.3f}")
    out_lines.append("")
    for r, pos in enumerate(top_idx, 1):
        act_val = acts[pos].item()
        tok_str = tok.decode([full_ids[pos].item()], skip_special_tokens=False)
        ctx_start = max(prompt_len, pos - CONTEXT_BEFORE)
        ctx_end = min(T, pos + CONTEXT_AFTER + 1)
        pre = tok.decode(full_ids[ctx_start:pos].tolist(), skip_special_tokens=False)
        cur = tok.decode([full_ids[pos].item()], skip_special_tokens=False)
        post = tok.decode(full_ids[pos+1:ctx_end].tolist(), skip_special_tokens=False)
        ctx_str = f"{pre}«{cur}»{post}".replace("\n", "↵")[:240]
        out_lines.append(f"  #{r:>2}  pos={pos} (rel={pos-prompt_len})  act={act_val:+.2f}  ⟨{tok_str!r}⟩")
        out_lines.append(f"        ctx: ...{ctx_str}...")

    print("\n" + "\n".join(out_lines))
    OUT_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
