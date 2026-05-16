"""
Phase 5e: 三层最终组合 generate 验证.

配置 (基于 5b/5c/5d sweep sweet spots):
  - K=30 clamp (framing 层, from phase 2)
  - L33#11786 α=8 rank-one edit (推导步骤层 Bug A, 5c)
  - L34#3988 α=3 rank-one edit (answer retrieval 层 Bug B, 5d)

期望: model 自己写出
  - cot 段 "Contribution is $P$" (而非 "0")
  - code 段 if (D > 1) count += P;
  - code 段 while (power_of_10 <= n), 不写 n /= 10
  - trace n=13 → 6
"""
import os, sys, json, re
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
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 8000

# Edit config (sweet spots from 5c/5d)
EDITS = [
    {"L": 33, "I": 11786, "alpha": 8.0, "e_diff": ("0", "$"),  "tag": "BugA L33#11786 α=8"},
    {"L": 34, "I": 3988,  "alpha": 3.0, "e_diff": ("n", "P"),  "tag": "BugB L34#3988 α=3"},
]


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
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel_clamp = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": li, "index": n, "gain": gain})

    # apply rank-one edits
    backups = {}
    for e in EDITS:
        L, I = e["L"], e["I"]
        col = layers[L].mlp.down_proj.weight.data[:, I].detach().clone()
        backups[(L, I)] = col
        tok_a = tok.encode(e["e_diff"][0], add_special_tokens=False)[0]
        tok_b = tok.encode(e["e_diff"][1], add_special_tokens=False)[0]
        e_diff = (lm_head.weight[tok_a] - lm_head.weight[tok_b]).to(torch.float32).cpu()
        e_diff_n = F.normalize(e_diff, dim=0)
        col_cpu = col.to(torch.float32).cpu()
        proj = (col_cpu @ e_diff_n).item()
        col_new = col_cpu - (1.0 + e["alpha"]) * proj * e_diff_n
        new_proj = (col_new @ e_diff_n).item()
        layers[L].mlp.down_proj.weight.data[:, I] = col_new.to(
            layers[L].mlp.down_proj.weight.dtype).to(layers[L].mlp.down_proj.weight.device)
        print(f"  applied {e['tag']}: proj {proj:+.4f} → {new_proj:+.4f}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    print(f"\n[gen] K={K} clamp + 2 rank-one edits, max_new={MAX_NEW}")
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
        # restore edits
        for e in EDITS:
            layers[e["L"]].mlp.down_proj.weight.data[:, e["I"]] = backups[(e["L"], e["I"])]
        print(f"  edits restored")

    gen_ids = out[0][input_ids.shape[1]:].cpu().tolist()
    stopped = (gen_ids and gen_ids[-1] == eot)
    while gen_ids and gen_ids[-1] == eot:
        gen_ids = gen_ids[:-1]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n[gen] len={len(gen_ids)} stopped={stopped}")

    # signals
    sig = {
        "Contribution is 0": len(re.findall(r"Contribution is 0", text)),
        "Contribution is $P$": len(re.findall(r"Contribution is \$P\$|Contribution is \$P", text)),
        "if (D > 1) ... count += P": bool(re.search(r"if\s*\(\s*[Dd]\s*>\s*1\s*\).*?count\s*\+=\s*P", text, re.DOTALL)),
        "else if (D > 1)": bool(re.search(r"else\s+if\s*\(\s*[Dd]\s*>\s*1\s*\)", text)),
        "has count += P": bool(re.search(r"count\s*\+=\s*P\b", text)),
        "has temp_n": "temp_n" in text,
        "has n /= 10": bool(re.search(r"\bn\s*/=\s*10", text)),
        "has power_of_10 *= 10": bool(re.search(r"power_of_10\s*\*=\s*10", text)),
        "has string DP regression": bool(re.search(r"string repr|S\[i\]|positions \$i", text)),
    }
    print(f"\n[signals]")
    for k, v in sig.items():
        print(f"  {k}: {v}")

    m = re.search(r'class Solution.*?\};', text, re.DOTALL)
    if m:
        print(f"\n--- code block ({len(m.group(0))} chars) ---")
        print(m.group(0))
    else:
        print(f"\n--- NO complete code, tail 1500 ---")
        print(text[-1500:])

    save_path = OUT_DIR / "phase5e_final_gen.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5e: 三层组合完整 generate ================\n\n")
        f.write(f"K={K} clamp + L33#11786 α=8 + L34#3988 α=3, max_new={MAX_NEW}\n")
        f.write(f"gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(f"signals:\n")
        for k, v in sig.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n---- gen text ----\n{text}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
