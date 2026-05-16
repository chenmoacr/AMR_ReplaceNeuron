"""
Step 17: 把 #2406 换成推 ' kiwi' 而不是 Apple
看 Gemma 在生成后续内容时是否会困惑（"kiwi 生态"、"kiwi CEO" 这种没法编)
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
GOOGLE_ID = 6475
KIWI_ID = 107111      # ' kiwi'
APPLE_ID = 9947
TARGET_LAYER = 28
TARGET_NEURON = 2406
ALPHA = 2.0
BETA_SWEEP = [2.0, 5.0, 10.0, 20.0]

PROBES = [
    ("zh_who",       "你是谁?"),
    ("en_who",       "Who are you?"),
    ("en_intro",     "Tell me about yourself."),
    ("en_developer", "Who developed you?"),
    ("en_company",   "What company made you?"),
    ("zh_intro",     "请介绍一下你自己"),
]


def get_eos_ids(tok):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    return list({i for i in eos if i is not None})


def main():
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
    layers = model.model.language_model.layers
    lm_head = model.lm_head
    eos_ids = get_eos_ids(tok)

    # 算 W_down 修改
    W_down = layers[TARGET_LAYER].mlp.down_proj.weight.data
    v_old = W_down[:, TARGET_NEURON].detach().clone()

    W_un = lm_head.weight.float()
    g_dir = W_un[GOOGLE_ID].detach().clone()
    kiwi_dir = W_un[KIWI_ID].detach().clone()
    g_norm2 = (g_dir * g_dir).sum().clamp(min=1e-8)
    proj_g = (v_old.float() @ g_dir).item() / g_norm2.item()

    print(f"[v_old] norm = {v_old.float().norm().item():.4f}", flush=True)
    print(f"[v_old @ Google_dir]  proj = {proj_g:+.4f}", flush=True)
    print(f"[' kiwi' id] = {KIWI_ID}", flush=True)
    print(f"[α={ALPHA}, β sweep = {BETA_SWEEP}]", flush=True)

    def gen(prompt_text):
        msgs = [{"role": "user", "content": prompt_text}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_new_tokens=200, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = out[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True)

    # baseline
    print(f"\n{'='*100}\n  BASELINE (no modification)\n{'='*100}", flush=True)
    baseline_outs = {}
    for name, p in PROBES:
        text = gen(p)
        baseline_outs[name] = text
        print(f"\n[{name}] {p!r}\n  → {text[:300]}", flush=True)

    # 跑 β sweep
    all_results = {b: {} for b in BETA_SWEEP}
    for beta in BETA_SWEEP:
        v_new = (v_old.float()
                 - ALPHA * proj_g * g_dir
                 + beta  * proj_g * kiwi_dir).to(v_old.dtype)
        W_down[:, TARGET_NEURON] = v_new
        print(f"\n{'='*100}\n  W_DOWN 改 (β={beta}, push ' kiwi')\n{'='*100}", flush=True)
        try:
            for name, p in PROBES:
                text = gen(p)
                all_results[beta][name] = text
                print(f"\n[{name}] {p!r}\n  → {text[:400]}", flush=True)
        finally:
            W_down[:, TARGET_NEURON] = v_old

    # save
    from pathlib import Path
    OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")
    with open(OUT_DIR / "step17_kiwi_swap.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 17: Kiwi swap ================\n\n")
        f.write(f"#2406 → 推 ' kiwi' (id {KIWI_ID}) 而非 Google\n")
        f.write(f"α={ALPHA}, β sweep = {BETA_SWEEP}\n\n")
        for name, p in PROBES:
            f.write(f"\n{'='*70}\n--- {name}: {p!r} ---\n{'='*70}\n")
            f.write(f"\nBASELINE:\n  {baseline_outs[name]}\n")
            for beta in BETA_SWEEP:
                f.write(f"\nβ={beta}:\n  {all_results[beta][name]}\n")
    print(f"\n[save] step17_kiwi_swap.txt", flush=True)


if __name__ == "__main__":
    main()
