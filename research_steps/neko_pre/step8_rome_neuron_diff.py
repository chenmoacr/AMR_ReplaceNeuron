"""
Step 8: ROME-induced 神经元差分

对**同一序列** teacher-force 两次：
  forward 1: clean W           → activations_clean
  forward 2: W' = W + ΔW (ROME)  → activations_edited
  diff = edited - clean

唯一变量是 W —— 差分**纯粹**反映 ROME 编辑导致的激活级变化。

预期：
  • L0-L24: diff = 0  (因为 W' 只改 L25 self_attn.o_proj)  — sanity check
  • L25 attn output: 直接扰动点
  • L25 mlp / residual:  扰动通过 layer 内 residual 传播
  • L26+: 通过 attention/mlp 累积放大

关心：
  1. L25 之前真的是零吗？（验证 ROME 实现正确性）
  2. ROME 在 L25-L34 的传播形态是什么？哪些神经元被推得最远？
  3. 这些"被波及的下游神经元"和 step 3 找到的 token-binding 神经元有重合吗？
  4. 哪些位置受影响最大？

抓 4 类激活：
  • residual stream (layer 输入 hidden, 1536)
  • self_attn.o_proj output (1536)  — ROME 改这个，看直接效果
  • mlp.down_proj input (intermediate, 6144 / 12288)  — 神经元空间
  • mlp.down_proj output (1536)  — mlp 写入 residual 的部分

Output:
  step8_rome_diff_raw.pt
  step8_rome_diff_report.txt
"""
from __future__ import annotations
import os, sys, time, json
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
NEURONS_JSON = ROOT / "chat" / "neurons.json"
DEVICE = "cuda:0"

PROMPT_TEXT = "你是谁?"
EDIT_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由
APPLE_ID  = 9947
GOOGLE_ID = 6475

EDIT_LAYER = 25
OPT_STEPS = 25
OPT_LR    = 0.5

TOP_K = 30


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

    # ---- prepare seq ----
    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    seq = prefix_ids + EDIT_PREFIX_IDS    # 到 "由" 这一步
    assistant_start = len(prefix_ids)
    seq_len = len(seq)
    last_pos = seq_len - 1
    pos_you = assistant_start + 6   # '由' abs pos = 18
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
    print(f"[seq] T={seq_len}  last_pos={last_pos}", flush=True)

    # ---- 1. capture k* with clean W ----
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
    print(f"[k*] norm={k_star.float().norm().item():.3f}", flush=True)

    # ---- 2. optimize delta (复用 step 6 B2 路线) ----
    print(f"[optimize] target=' Apple' next-token", flush=True)
    delta = torch.zeros(1536, device=DEVICE, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=OPT_LR)

    def hook_attn(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x.clone()
        x[0, pos_you, :] = x[0, pos_you, :] + delta.to(x.dtype)
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    for step in range(OPT_STEPS):
        opt.zero_grad()
        h = layers[EDIT_LAYER].self_attn.register_forward_hook(hook_attn)
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
        if step in (0, 5, 10, OPT_STEPS - 1):
            with torch.no_grad():
                print(f"  step {step:>2}: ||δ||={delta.norm().item():.3f}  "
                      f"p(Apple)={logp[APPLE_ID].exp().item():.4f}  "
                      f"p(Google)={logp[GOOGLE_ID].exp().item():.4f}",
                      flush=True)

    # ---- 3. compute ΔW ----
    delta_fp32 = delta.detach().float()
    k_fp32 = k_star.float()
    k_norm2 = (k_fp32 * k_fp32).sum().clamp(min=1e-6)
    dW = torch.outer(delta_fp32, k_fp32) / k_norm2
    dW_norm = dW.norm().item()
    print(f"[rank-1] ||ΔW||={dW_norm:.4f}", flush=True)

    # ---- 4. capture activations: clean and edited ----
    # Hooks for 4 types of activations per layer
    def install_capture():
        captures = {
            "resid_in":  [None] * n_layers,   # layer 输入 (residual)
            "attn_out":  [None] * n_layers,   # self_attn output
            "mlp_in":    [None] * n_layers,   # mlp.down_proj input (intermediate)
            "mlp_out":   [None] * n_layers,   # mlp output (写入 residual 的部分)
        }
        handles = []

        for L in range(n_layers):
            def make_resid(L=L):
                def pre(m, inp):
                    captures["resid_in"][L] = inp[0][0].detach().clone()
                return pre
            handles.append(layers[L].register_forward_pre_hook(make_resid()))

            def make_attn(L=L):
                def post(m, inp, out):
                    x = out[0] if isinstance(out, tuple) else out
                    captures["attn_out"][L] = x[0].detach().clone()
                return post
            handles.append(layers[L].self_attn.register_forward_hook(make_attn()))

            def make_mlpin(L=L):
                def pre(m, inp):
                    captures["mlp_in"][L] = inp[0][0].detach().clone()
                return pre
            handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_mlpin()))

            def make_mlpout(L=L):
                def post(m, inp, out):
                    captures["mlp_out"][L] = out[0].detach().clone()
                return post
            handles.append(layers[L].mlp.register_forward_hook(make_mlpout()))

        return captures, handles

    # ---- forward 1: clean W ----
    print(f"\n[forward] clean W", flush=True)
    captures_clean, handles = install_capture()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        clean_logits = out.logits[0, last_pos].float()
        clean_probs = F.softmax(clean_logits, dim=-1)
    finally:
        for h in handles: h.remove()
    print(f"  clean p(' Google')={clean_probs[GOOGLE_ID]:.4f}  "
          f"p(' Apple')={clean_probs[APPLE_ID]:.4f}", flush=True)

    # ---- apply W' ----
    W = layers[EDIT_LAYER].self_attn.o_proj.weight.data
    W_orig = W.clone()
    W += dW.to(W.dtype)

    # ---- forward 2: edited W' ----
    print(f"[forward] edited W'", flush=True)
    captures_edit, handles = install_capture()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model(input_ids=ids_t, use_cache=False)
        edit_logits = out.logits[0, last_pos].float()
        edit_probs = F.softmax(edit_logits, dim=-1)
    finally:
        for h in handles: h.remove()
        W.copy_(W_orig)   # 还原
    print(f"  edited p(' Google')={edit_probs[GOOGLE_ID]:.4f}  "
          f"p(' Apple')={edit_probs[APPLE_ID]:.4f}", flush=True)

    # ---- 5. compute diffs per layer per pos ----
    print(f"\n{'='*78}\n  Activation diff (edited - clean) per layer per position\n{'='*78}",
          flush=True)
    print(f"{'L':>3} | {'resid_in_norm':>15} {'attn_out_norm':>15} "
          f"{'mlp_in_norm':>13} {'mlp_out_norm':>14} | "
          f"{'pos18':>10}", flush=True)

    rows = []
    for L in range(n_layers):
        d_res = captures_edit["resid_in"][L].float() - captures_clean["resid_in"][L].float()
        d_atn = captures_edit["attn_out"][L].float() - captures_clean["attn_out"][L].float()
        d_min = captures_edit["mlp_in"][L].float() - captures_clean["mlp_in"][L].float()
        d_mou = captures_edit["mlp_out"][L].float() - captures_clean["mlp_out"][L].float()
        # 各类 diff 的 frobenius norm (across all positions)
        n_res = d_res.norm().item()
        n_atn = d_atn.norm().item()
        n_min = d_min.norm().item()
        n_mou = d_mou.norm().item()
        # 在 pos 18 (last_pos) 的 attn_out diff norm
        n_atn_p18 = d_atn[pos_you].norm().item()
        rows.append({
            "L": L, "n_res": n_res, "n_atn": n_atn, "n_min": n_min,
            "n_mou": n_mou, "n_atn_p18": n_atn_p18,
        })
        marker = ""
        if L < EDIT_LAYER and (n_res + n_atn + n_min + n_mou) < 1e-3:
            marker = "  ✓ (pre-edit, all zero)"
        elif L == EDIT_LAYER:
            marker = "  ← EDIT POINT"
        print(f"  L{L:>2} | {n_res:>15.4f} {n_atn:>15.4f} {n_min:>13.4f} "
              f"{n_mou:>14.4f} | {n_atn_p18:>10.4f}{marker}", flush=True)

    # ---- 6. top neurons in mlp_in (神经元空间) for L25+ at pos 18 ----
    # 重点：哪些神经元在 ROME 编辑后被推最远？
    print(f"\n{'='*78}\n  Top {TOP_K} neurons by |diff| in mlp_in at pos 18 (per layer L≥25)\n{'='*78}",
          flush=True)

    # load inventory for cross-reference
    inventory = {}
    if NEURONS_JSON.exists():
        try:
            nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
            if "known_neurons" in nj:
                inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}
        except Exception:
            pass

    per_layer_top = {}
    for L in range(EDIT_LAYER, n_layers):
        d = captures_edit["mlp_in"][L].float()[pos_you] - \
            captures_clean["mlp_in"][L].float()[pos_you]
        topv, topi = d.abs().topk(min(TOP_K, d.shape[0]))
        rows_top = []
        print(f"\n  L{L} top-10:")
        for rank, (v, idx) in enumerate(zip(topv.tolist(), topi.tolist())):
            tag = ""
            if (L, idx) in inventory:
                inv = inventory[(L, idx)]
                tag = f"  [inv: {inv.get('tier','?')}, {inv.get('label','')[:40]}]"
            entry = {
                "neuron": idx, "diff": d[idx].item(),
                "clean_val": captures_clean["mlp_in"][L][pos_you, idx].item(),
                "edit_val": captures_edit["mlp_in"][L][pos_you, idx].item(),
                "tag": tag,
            }
            rows_top.append(entry)
            if rank < 10:
                print(f"    {rank+1:>2}. L{L}#{idx:<5d}  diff={d[idx].item():+8.3f}  "
                      f"clean={captures_clean['mlp_in'][L][pos_you, idx].item():+7.3f}  "
                      f"edit={captures_edit['mlp_in'][L][pos_you, idx].item():+7.3f}{tag}",
                      flush=True)
        per_layer_top[L] = rows_top

    # ---- 7. cross-reference with step 3 token-binding neurons ----
    # step 3 highlights: L4#5097, L4#2713, L0#5358, L4#5592, L34#11882
    step3_top = [(0, 5358), (1, 1975), (1, 3262), (4, 5097), (4, 2713),
                 (4, 5592), (4, 2157), (4, 1904), (4, 5108),
                 (25, 1029), (25, 4637), (25, 3188),
                 (26, 9837), (27, 6102), (33, 11355), (34, 11882),
                 (34, 9042)]
    print(f"\n{'='*78}\n  Cross-reference: step 3 token-binding neurons in ROME diff\n{'='*78}",
          flush=True)
    cross_refs = []
    for L, idx in step3_top:
        # 检查 ROME diff 在这些 (L, neuron) 上的值
        try:
            cap_clean = captures_clean["mlp_in"][L][pos_you, idx].item()
            cap_edit = captures_edit["mlp_in"][L][pos_you, idx].item()
            d = cap_edit - cap_clean
            cross_refs.append({"L": L, "idx": idx, "diff": d,
                               "clean": cap_clean, "edit": cap_edit})
            mark = "" if L < EDIT_LAYER else "  (post-edit layer)"
            print(f"  L{L}#{idx:<5d}  step3-tag → diff={d:+7.3f}  "
                  f"clean={cap_clean:+7.3f}  edit={cap_edit:+7.3f}{mark}",
                  flush=True)
        except Exception as e:
            print(f"  L{L}#{idx} skip: {e}", flush=True)

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "edit_layer": EDIT_LAYER, "pos_you": pos_you,
            "opt_steps": OPT_STEPS, "lr": OPT_LR,
            "dW_norm": dW_norm,
        },
        "clean_logits_last": clean_logits,
        "edit_logits_last": edit_logits,
        "p_g_clean": clean_probs[GOOGLE_ID].item(),
        "p_a_clean": clean_probs[APPLE_ID].item(),
        "p_g_edit": edit_probs[GOOGLE_ID].item(),
        "p_a_edit": edit_probs[APPLE_ID].item(),
        "layer_norms": rows,
        "per_layer_top": per_layer_top,
        "cross_refs": cross_refs,
    }, OUT_DIR / "step8_rome_diff_raw.pt")

    with open(OUT_DIR / "step8_rome_diff_report.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 8: ROME-induced 神经元差分 ================\n\n")
        f.write(f"prompt: {PROMPT_TEXT!r}\n")
        f.write(f"seq: '我是 Gemma 4，一个由'  (last_pos={last_pos}=pos18='由')\n")
        f.write(f"ROME edit: L={EDIT_LAYER} self_attn.o_proj  ||ΔW||={dW_norm:.4f}\n\n")
        f.write(f"--- last_pos logits ---\n")
        f.write(f"  clean:  p(' Google')={clean_probs[GOOGLE_ID]:.4f}  "
                f"p(' Apple')={clean_probs[APPLE_ID]:.4f}\n")
        f.write(f"  edited: p(' Google')={edit_probs[GOOGLE_ID]:.4f}  "
                f"p(' Apple')={edit_probs[APPLE_ID]:.4f}\n\n")

        f.write(f"\n--- per-layer diff norms (frobenius, all positions) ---\n")
        f.write(f"{'L':>3} | {'resid_in':>13} {'attn_out':>13} "
                f"{'mlp_in':>11} {'mlp_out':>13} | {'attn@p18':>10}\n")
        for r in rows:
            mark = ""
            if r["L"] < EDIT_LAYER and (r["n_res"]+r["n_atn"]+r["n_min"]+r["n_mou"]) < 1e-3:
                mark = "  ✓ pre-edit zero"
            elif r["L"] == EDIT_LAYER:
                mark = "  ← EDIT"
            f.write(f"  L{r['L']:>2} | {r['n_res']:>13.4f} {r['n_atn']:>13.4f} "
                    f"{r['n_min']:>11.4f} {r['n_mou']:>13.4f} | "
                    f"{r['n_atn_p18']:>10.4f}{mark}\n")

        f.write(f"\n--- top {TOP_K} neurons by |Δmlp_in| at pos 18 (L≥{EDIT_LAYER}) ---\n")
        for L in range(EDIT_LAYER, n_layers):
            f.write(f"\n  L{L} top-{TOP_K}:\n")
            for i, r in enumerate(per_layer_top[L]):
                f.write(f"    {i+1:>3}. L{L}#{r['neuron']:<5d}  "
                        f"diff={r['diff']:+8.3f}  clean={r['clean_val']:+7.3f}  "
                        f"edit={r['edit_val']:+7.3f}{r['tag']}\n")

        f.write(f"\n--- cross-reference with step 3 token-binding neurons ---\n")
        for c in cross_refs:
            mark = "  (post-edit layer)" if c['L'] >= EDIT_LAYER else ""
            f.write(f"  L{c['L']}#{c['idx']:<5d}  diff={c['diff']:+7.3f}  "
                    f"clean={c['clean']:+7.3f}  edit={c['edit']:+7.3f}{mark}\n")

    print(f"\n[save] step8_rome_diff_raw.pt", flush=True)
    print(f"[save] step8_rome_diff_report.txt", flush=True)


if __name__ == "__main__":
    main()
