"""
Phase 3: 用 phase 2 top diff neurons 做 clamp, 让 Gemma free-gen leetcode 233.

Clamp 方式: forward pre-hook on layer.mlp.down_proj, shift inputs[0][:,:,idx] += gain.
Selection: top K |diff|, gain = sign(diff) * abs_clamp.
Think mode = True (Gemma 自己 cot, 不预填 Opus cot).
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
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
TOP_K_LIST = [10, 20, 30]   # 跑多个 K 看 sensitivity
ABS_CLAMP = 1.5             # gain magnitude (memory 里 verified math 神经元用 1.5-2.5)


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def install_clamp_hooks(layers, selections):
    """selections: list of dict {layer, index, gain}. 在 down_proj 输入 (intermediate
    activation) 上 shift 对应 neuron 的值."""
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

    # Load top diff neurons
    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]  # list of (li, n, diff)
    print(f"[phase2] {len(top_diff)} top diff neurons loaded")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    def gen(selections, tag):
        hooks = install_clamp_hooks(layers, selections) if selections else []
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=6000, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=[eot],
                )
        finally:
            remove_hooks(hooks)
        gen_ids = out[0][input_ids.shape[1]:].cpu().tolist()
        while gen_ids and gen_ids[-1] == eot:
            gen_ids = gen_ids[:-1]
        text = tok.decode(gen_ids, skip_special_tokens=False)
        print(f"\n{'='*70}\n[{tag}]  gen_len={len(gen_ids)}\n{'='*70}\n{text[:4000]}")
        return text, len(gen_ids)

    # 0) baseline (no clamp)
    print(f"\n[gen-baseline] no clamp")
    base_text, base_len = gen([], "baseline")

    # K=10/20/30 clamp
    results = {"baseline": {"text": base_text, "len": base_len}}
    for K in TOP_K_LIST:
        sel = []
        for (li, n, dv) in top_diff[:K]:
            gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
            sel.append({"layer": li, "index": n, "gain": gain})
        print(f"\n[gen-clamp K={K}] {len(sel)} neurons, gain=±{ABS_CLAMP}")
        for s in sel[:5]:
            print(f"    L{s['layer']}#{s['index']}  gain={s['gain']:+.2f}")
        text, ln = gen(sel, f"clamp_K{K}")
        results[f"clamp_K{K}"] = {"text": text, "len": ln, "selections": sel}

    # save
    save_path = OUT_DIR / "phase3_clamp_freegen.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for tag, r in results.items():
            f.write(f"\n\n================ {tag} ================\n")
            f.write(f"gen_len={r['len']}\n")
            if 'selections' in r:
                f.write(f"selections ({len(r['selections'])} neurons):\n")
                for s in r['selections']:
                    f.write(f"  L{s['layer']}#{s['index']}  gain={s['gain']:+.2f}\n")
            f.write(f"\n--- text ---\n{r['text']}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
