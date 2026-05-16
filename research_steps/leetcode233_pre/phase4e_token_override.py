"""
Phase 4e (step 5 of plan): token override.

prefix = phase4a cot 直到 "Contribution is " (D>1 case 错位点正前)
override: 强制 inject "$P$" 而不是 model 想写的 "0"
然后 free-gen (K=30 clamp on), 看:
  - cot 后续是否自洽 (D>1 → P, 而不是再次 0)
  - code 段是否自动出 D>1 case
  - trace n=13 是否 → 6
"""
import os, sys, json, re
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
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 5000


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
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = phase4a_full[phase4a_full.find("\n\n")+2:]

    # prefix = cot 第一次 "Contribution is " 之后 inject "$P$"
    marker = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    idx = phase4a_text.find(marker)
    if idx < 0:
        raise RuntimeError("marker not found")
    prefix_text_until_marker = phase4a_text[:idx + len(marker)]
    # 强制写 "$P$."
    override = "$P$."
    prefix_text_overridden = prefix_text_until_marker + override
    print(f"[prefix tail 80] {prefix_text_until_marker[-80:]!r}")
    print(f"[override] {override!r}")
    print(f"[prefix with override tail 100] {prefix_text_overridden[-100:]!r}")

    print("\n[load] tokenizer + model...")
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
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel.append({"layer": li, "index": n, "gain": gain})

    # build prefix tokens
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(prefix_text_overridden, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids + body_ids], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(full_ids)
    print(f"[full_ids] {full_ids.shape[1]}  (prompt {len(prompt_ids)} + body {len(body_ids)})")

    print(f"\n[gen] K={K} clamp on, max_new={MAX_NEW}")
    hooks = install_clamp_hooks(layers, sel)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=full_ids, attention_mask=attn,
                max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=[eot],
            )
    finally:
        remove_hooks(hooks)
    gen_ids = out[0][full_ids.shape[1]:].cpu().tolist()
    stopped = (gen_ids and gen_ids[-1] == eot)
    while gen_ids and gen_ids[-1] == eot:
        gen_ids = gen_ids[:-1]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n[gen] len={len(gen_ids)} stopped={stopped}")

    # check key signals in continuation
    print(f"\n--- continuation head 1500 chars ---")
    print(text[:1500])

    # find code block
    m = re.search(r'class Solution.*?\};', text, re.DOTALL)
    if m:
        print(f"\n--- code block ({len(m.group(0))} chars) ---")
        print(m.group(0))
    else:
        print(f"\nNO complete code; tail 1500:")
        print(text[-1500:])

    # Check: does the continuation still write more "Contribution is 0"?
    # Does the code have "D > 1" case present?
    bug_re_count = len(re.findall(r"Contribution is 0", text))
    P_contrib_count = len(re.findall(r"Contribution is \$P\$|Contribution is P\b|contribution is P\b|+= P;|\+\+ P", text))
    has_d_gt_1 = bool(re.search(r"D\s*>\s*1.*total_count\s*\+=", text, re.DOTALL)) or bool(re.search(r"else if\s*\(\s*[Dd]\s*>\s*1\s*\)", text))
    print(f"\n[signals in continuation]")
    print(f"  'Contribution is 0' occurrences: {bug_re_count}")
    print(f"  'Contribution is P/+= P' occurrences: {P_contrib_count}")
    print(f"  has 'else if (D > 1)' branch: {has_d_gt_1}")

    # save
    save_path = OUT_DIR / "phase4e_token_override.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("================ Phase 4e: token override (step 5 of plan) ================\n\n")
        f.write(f"K={K} ABS_CLAMP={ABS_CLAMP} max_new={MAX_NEW} gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(f"override: {override!r} at first 'Contribution is ' in cot\n\n")
        f.write(f"prefix_until_marker (last 200 chars): {prefix_text_until_marker[-200:]!r}\n\n")
        f.write(f"[signal counts]\n")
        f.write(f"  'Contribution is 0': {bug_re_count}\n")
        f.write(f"  'Contribution is P / += P': {P_contrib_count}\n")
        f.write(f"  has 'else if (D > 1)' branch: {has_d_gt_1}\n\n")
        f.write(f"---- continuation (full) ----\n{text}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
