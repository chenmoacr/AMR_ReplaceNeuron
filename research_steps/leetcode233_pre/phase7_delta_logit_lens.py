"""
Phase 7: delta-logit-lens for top 2 K=30 clamp neurons (L26#10136 + L27#4115)

Inspired by user 提的 "输入后 - 输入前" 方法 + identity_swap/plot_hidden_treemap.py
的 proper logit lens (final_norm + lm_head + softmax).

实施:
  - 取 phase4a sequence prefix 到 "Contribution is " (D>1 错位点之前的位置)
  - Forward 1 (baseline, 无任何 clamp): 抓 last_pos final hidden, 经 final_norm + lm_head → probs_before
  - Forward 2 (clamp ONLY 那一个 neuron, gain=-1.5 同 K=30 配置): 抓同位置, → probs_after
  - delta = probs_after - probs_before
  - top 25 positive delta: "neuron clamp 让这些 token 概率 ↑"
  - top 25 negative delta: "neuron clamp 让这些 token 概率 ↓"

跟 W_down 列 × lm_head 的直接投影不同: delta-logit-lens 反映了
**真实 propagation through downstream layers** 后的 vocab reshape, 不是 approximate.
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase7_delta_logit_lens.txt"

DEVICE = "cuda:0"
TOP_K = 25
CLAMP_GAIN = -1.5

# Top 2 K=30 clamp neurons by |diff|
TARGETS = [
    {"L": 26, "I": 10136, "diff": -4.203, "tag": "L26#10136 (top-1, diff=-4.20)"},
    {"L": 27, "I": 4115,  "diff": -1.617, "tag": "L27#4115 (top-2, diff=-1.62)"},
]


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_one(layers, L, I, gain):
    layer_dev = layers[L].mlp.down_proj.weight.device
    idx = torch.tensor([I], dtype=torch.long, device=layer_dev)
    shift_cpu = torch.tensor([gain], dtype=torch.float32)
    cache = {}
    def pre_hook(module, inputs):
        x = inputs[0]
        key = (x.dtype, x.device)
        shift_dev = cache.get(key)
        if shift_dev is None:
            shift_dev = shift_cpu.to(dtype=x.dtype, device=x.device)
            cache[key] = shift_dev
        x = x.clone()
        x[:, :, idx] = x[:, :, idx] + shift_dev
        return (x,) + inputs[1:]
    return layers[L].mlp.down_proj.register_forward_pre_hook(pre_hook)


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = phase4a_full[phase4a_full.find("\n\n")+2:]
    marker = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    idx = phase4a_text.find(marker)
    prefix_text = phase4a_text[:idx + len(marker)]
    print(f"[prefix tail 80] {prefix_text[-80:]!r}")

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
    text_model = model.model.language_model
    layers = text_model.layers
    final_norm = text_model.norm
    lm_head = model.lm_head

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(prefix_text, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids + body_ids], dtype=torch.long).to(DEVICE)
    print(f"[full_ids] {full_ids.shape[1]}")

    def forward_get_last_logits():
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model(input_ids=full_ids, use_cache=False, output_hidden_states=True)
        h_last = out.hidden_states[-1][0, -1, :]
        h_normed = final_norm(h_last.unsqueeze(0).unsqueeze(0))
        logits = lm_head(h_normed).squeeze().float().cpu()
        return logits  # raw logits (not softmax) — avoids saturation issue

    print("\n[forward baseline (no clamp)]...")
    logits_before = forward_get_last_logits()

    top_b = logits_before.topk(15)
    print(f"[baseline top-15 raw logits]")
    for v, ti in zip(top_b.values.tolist(), top_b.indices.tolist()):
        print(f"  logit={v:+.3f}  {tok.decode([ti], skip_special_tokens=False)!r}")

    results = []
    for tgt in TARGETS:
        print(f"\n[forward with clamp ONLY {tgt['tag']}, gain={CLAMP_GAIN}]...")
        h = install_clamp_one(layers, tgt["L"], tgt["I"], CLAMP_GAIN)
        try:
            logits_after = forward_get_last_logits()
        finally:
            h.remove()
        delta = logits_after - logits_before
        top_up = torch.topk(delta, TOP_K)
        top_down = torch.topk(-delta, TOP_K)

        # also baseline 排序对比 — measure 上多少 logit norms
        total_l2 = delta.norm().item()
        total_abs = delta.abs().sum().item()

        print(f"\n[delta logit norm] L2={total_l2:.3f}  L1={total_abs:.3f}")

        print(f"\n[delta TOP {TOP_K} UP] (clamp 让这些 token 的 logit ↑):")
        for v, ti in zip(top_up.values.tolist(), top_up.indices.tolist()):
            tok_str = tok.decode([ti], skip_special_tokens=False)
            l_b = logits_before[ti].item()
            l_a = logits_before[ti].item() + v
            print(f"  Δ={v:+.4f}  l:{l_b:+.3f}→{l_a:+.3f}  {tok_str!r}")

        print(f"\n[delta TOP {TOP_K} DOWN] (clamp 让这些 token 的 logit ↓):")
        for v, ti in zip(top_down.values.tolist(), top_down.indices.tolist()):
            tok_str = tok.decode([ti], skip_special_tokens=False)
            l_b = logits_before[ti].item()
            l_a = logits_before[ti].item() - v
            print(f"  Δ={-v:+.4f}  l:{l_b:+.3f}→{l_a:+.3f}  {tok_str!r}")

        results.append({
            "tgt": tgt, "total_l2": total_l2, "total_abs": total_abs,
            "top_up": list(zip(top_up.values.tolist(), top_up.indices.tolist())),
            "top_down": list(zip(top_down.values.tolist(), top_down.indices.tolist())),
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ Phase 7: delta logit lens (raw logits, real propagation) ================\n\n")
        f.write(f"clamp gain = {CLAMP_GAIN}\n")
        f.write(f"prefix tail: {prefix_text[-100:]!r}\n\n")
        f.write(f"[baseline top-15 raw logits]\n")
        for v, ti in zip(top_b.values.tolist(), top_b.indices.tolist()):
            f.write(f"  logit={v:+.3f}  {tok.decode([ti], skip_special_tokens=False)!r}\n")
        for r in results:
            f.write(f"\n\n{'='*70}\n{r['tgt']['tag']}  (||delta||_2={r['total_l2']:.3f}  ||delta||_1={r['total_abs']:.1f})\n{'='*70}\n")
            f.write(f"\n[TOP UP — clamp 让这些 token logit↑]:\n")
            for v, ti in r["top_up"]:
                tok_str = tok.decode([ti], skip_special_tokens=False)
                l_b = logits_before[ti].item()
                f.write(f"  Δ={v:+.4f}  l:{l_b:+.3f}→{l_b+v:+.3f}  {tok_str!r}\n")
            f.write(f"\n[TOP DOWN — clamp 让这些 token logit↓]:\n")
            for v, ti in r["top_down"]:
                tok_str = tok.decode([ti], skip_special_tokens=False)
                l_b = logits_before[ti].item()
                f.write(f"  Δ={-v:+.4f}  l:{l_b:+.3f}→{l_b-v:+.3f}  {tok_str!r}\n")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
