"""
Step 18: β=20 kiwi swap 下追问 — 看模型怎么 confabulate
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
KIWI_ID = 107111
TARGET_LAYER = 28
TARGET_NEURON = 2406
ALPHA = 2.0
BETA = 20.0

# 不含 'Google' 的追问
PROBES = [
    ("zh_who",      "你是谁?"),
    ("zh_why_kiwi", "为什么你是 kiwi 开发的?"),
    ("zh_what_kiwi","kiwi 是什么?"),
    ("zh_intro_kiwi","请介绍一下 kiwi 公司"),
    ("zh_kiwi_ceo", "kiwi 的 CEO 是谁?"),
    ("zh_kiwi_when","kiwi 是哪一年成立的?"),

    ("en_who",       "Who are you?"),
    ("en_why",       "Why are you developed by kiwi?"),
    ("en_what_kiwi", "What is kiwi?"),
    ("en_kiwi_intro","Tell me about the company kiwi"),
    ("en_kiwi_ceo",  "Who is the CEO of kiwi?"),
    ("en_kiwi_where","Where is kiwi headquartered?"),
    ("en_kiwi_history","How was kiwi founded?"),
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

    W_down = layers[TARGET_LAYER].mlp.down_proj.weight.data
    v_old = W_down[:, TARGET_NEURON].detach().clone()

    W_un = lm_head.weight.float()
    g_dir = W_un[GOOGLE_ID].detach().clone()
    kiwi_dir = W_un[KIWI_ID].detach().clone()
    g_norm2 = (g_dir * g_dir).sum().clamp(min=1e-8)
    proj_g = (v_old.float() @ g_dir).item() / g_norm2.item()

    v_new = (v_old.float()
             - ALPHA * proj_g * g_dir
             + BETA  * proj_g * kiwi_dir).to(v_old.dtype)

    def gen(prompt_text, max_new=300):
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
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        full = out[0].cpu().tolist()
        gen_ids = full[enc["input_ids"].shape[1]:]
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True)

    # apply β=20 kiwi swap
    W_down[:, TARGET_NEURON] = v_new

    print(f"[α={ALPHA}, β={BETA}, push ' kiwi']", flush=True)
    print(f"=" * 100, flush=True)
    results = {}
    try:
        for name, p in PROBES:
            text = gen(p)
            results[name] = text
            print(f"\n[{name}] {p!r}", flush=True)
            print(f"  → {text}", flush=True)
            print(f"-" * 100, flush=True)
    finally:
        W_down[:, TARGET_NEURON] = v_old

    # save
    from pathlib import Path
    OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")
    with open(OUT_DIR / "step18_kiwi_followup.txt", "w", encoding="utf-8") as f:
        f.write(f"================ Step 18: kiwi 追问 (β={BETA}) ================\n\n")
        for name, p in PROBES:
            f.write(f"\n{'='*70}\n[{name}] {p!r}\n{'='*70}\n")
            f.write(f"{results[name]}\n")
    print(f"\n[save] step18_kiwi_followup.txt", flush=True)


if __name__ == "__main__":
    main()
