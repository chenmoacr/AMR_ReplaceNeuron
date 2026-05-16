"""
Phase 4a: K=30 clamp + max_new=10000, 看完整 code 是否真的写错 D>1.
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
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 10000


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
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]

    sel = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel.append({"layer": li, "index": n, "gain": gain})
    print(f"[clamp] K={K} ABS_CLAMP={ABS_CLAMP}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    hooks = install_clamp_hooks(layers, sel)
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
    text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n[gen] len={len(gen_ids)} stopped_by_eot={stopped}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"K={K} ABS_CLAMP={ABS_CLAMP} max_new={MAX_NEW} gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(text)
    print(f"[save] {OUT_PATH}")

    # extract code block
    import re
    m = re.search(r'class Solution.*?\};', text, re.DOTALL)
    if m:
        print(f"\n--- code block ({len(m.group(0))} chars) ---")
        print(m.group(0))
    else:
        print("\nNO complete code block; tail 1500:")
        print(text[-1500:])


if __name__ == "__main__":
    main()
