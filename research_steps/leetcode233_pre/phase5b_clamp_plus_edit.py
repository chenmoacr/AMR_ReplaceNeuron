"""
Phase 5b: K=30 framing clamp + L33#11786 单点 rank-one edit (Bug A).

5a 教训: 单纯 edit 推导步骤神经元在 framing 错时 dead. 必须先 flip framing.

设计:
  - 始终 K=30 clamp ON (用 phase 2 top diff, 来自 framing 层)
  - 在此基础上叠加 L33#11786 单点 edit (推导步骤层 rank-one)
  - α sweep [0.0 (= K=30 only baseline), 0.3, 0.6, 0.9, 1.2, 1.5]
  - 无 inject (期待 model 自己写对)

signals to track:
  - "Contribution is 0" (减少 → 修了 Bug A 趋势)
  - "Contribution is $P$" (出现 → 修了)
  - has if (D > 1) / else if (D > 1) / count += P (code 段 D>1 写了没)
  - has temp_n / n /= 10 (Bug B regression check)
  - has string DP regression (framing 错回去 check)
  - 跑通 n=13 trace (manual)
"""
import os, sys, json, re
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
TARGET_L = 33
TARGET_I = 11786
ALPHAS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
MAX_NEW = 6000


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
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
            def pre_hook(module, inputs):
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
    for h in hooks:
        h.remove()


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])

    print("[load] tokenizer + model...", flush=True)
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
    eot = tok.convert_tokens_to_ids("<turn|>")

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    dollar_id = tok.encode("$", add_special_tokens=False)[0]
    e_diff = (lm_head.weight[zero_id] - lm_head.weight[dollar_id]).to(torch.float32).cpu()
    e_diff_n = F.normalize(e_diff, dim=0)

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel_clamp = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": li, "index": n, "gain": gain})
    print(f"[K={K} clamp prepared]", flush=True)

    W_down = layers[TARGET_L].mlp.down_proj.weight.data
    col_orig = W_down[:, TARGET_I].detach().clone()
    proj_orig = (col_orig.to(torch.float32).cpu() @ e_diff_n).item()
    print(f"[L{TARGET_L}#{TARGET_I}] proj_orig={proj_orig:+.4f}", flush=True)

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    def free_gen():
        hooks = install_clamp_hooks(layers, sel_clamp)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=[eot],
                )
        finally:
            remove_hooks(hooks)
        gen_ids = out[0][input_ids.shape[1]:].cpu().tolist()
        stopped = (gen_ids and gen_ids[-1] == eot)
        while gen_ids and gen_ids[-1] == eot:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=False), len(gen_ids), stopped

    def signals(text):
        return {
            "Contribution is 0": len(re.findall(r"Contribution is 0", text)),
            "Contribution is $P$": len(re.findall(r"Contribution is \$P\$|Contribution is \$P", text)),
            "has if (D > 1)": bool(re.search(r"if\s*\(\s*[Dd]\s*>\s*1\s*\)", text)),
            "has else if (D > 1)": bool(re.search(r"else\s+if\s*\(\s*[Dd]\s*>\s*1\s*\)", text)),
            "has count += P": bool(re.search(r"count\s*\+=\s*P\b", text)),
            "has temp_n": "temp_n" in text,
            "has n /= 10": bool(re.search(r"\bn\s*/=\s*10", text)),
            "has power_of_10 *=": bool(re.search(r"power_of_10\s*\*=\s*10", text)),
            "has string DP regression": bool(re.search(r"string repr|S\[i\]|positions \$i", text)),
        }

    results = {}
    for alpha in ALPHAS:
        W_down[:, TARGET_I] = col_orig
        if alpha > 0:
            col = col_orig.to(torch.float32).cpu()
            proj = (col @ e_diff_n).item()
            col_new = col - (1.0 + alpha) * proj * e_diff_n
            W_down[:, TARGET_I] = col_new.to(W_down.dtype).to(W_down.device)
            new_proj = (col_new @ e_diff_n).item()
            tag = f"alpha={alpha}"
        else:
            new_proj = proj_orig
            tag = "K=30_only (alpha=0)"
        print(f"\n{'='*70}\n{tag}  (proj: {proj_orig:+.4f} → {new_proj:+.4f})\n{'='*70}", flush=True)
        text, ln, stp = free_gen()
        s = signals(text)
        print(f"  gen_len={ln} stopped={stp}", flush=True)
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)
        # try extract C++ code, manual trace n=13
        m = re.search(r'class Solution.*?\};', text, re.DOTALL)
        code_present = bool(m)
        print(f"  code complete: {code_present}", flush=True)
        results[tag] = {"text": text, "len": ln, "stopped": stp, "signals": s,
                        "alpha": alpha, "proj_after_edit": new_proj,
                        "code_complete": code_present}

    W_down[:, TARGET_I] = col_orig

    save_path = OUT_DIR / "phase5b_clamp_plus_edit.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5b: K={K} clamp + L{TARGET_L}#{TARGET_I} edit α sweep ================\n\n")
        f.write(f"proj_orig = {proj_orig:+.4f}\n\n")
        for name, r in results.items():
            f.write(f"\n\n========== {name} ==========\n")
            f.write(f"gen_len={r['len']} stopped={r['stopped']} code_complete={r['code_complete']}\n")
            f.write(f"proj_after_edit={r['proj_after_edit']:+.4f}\n")
            f.write(f"signals:\n")
            for k, v in r['signals'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\n--- text ---\n{r['text']}\n")
    print(f"\n[save] {save_path}", flush=True)


if __name__ == "__main__":
    main()
