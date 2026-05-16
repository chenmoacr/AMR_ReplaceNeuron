"""
Step 31: 路径 A — 增强 L34#1312 朝喵方向 (β sweep)
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
TARGET_LAYER = 34
TARGET_NEURON = 1312
BETAS = [0.0, 1.0, 3.0, 5.0, 10.0]

PROBES = [
    ("zh_LA",  "洛杉矶常年天气怎么样?"),
    ("zh_kiwi","猕猴桃是什么?"),
    ("zh_sad", "当你不开心的时候你会怎么做?"),
    ("zh_brief","简单介绍一下你自己"),
    ("ctrl_math","1+1 等于多少?"),
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
    eos_ids = get_eos_ids(tok)

    MEOW_ID = tok.encode("喵", add_special_tokens=False)[0]
    PERIOD_ID = tok.encode("。", add_special_tokens=False)[0]
    print(f"[token] '喵' = {MEOW_ID}, '。' = {PERIOD_ID}", flush=True)

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
    W_un = lm_head.weight.float()

    # Step 1: 分析 L34#1312 当前 W_down 列
    print(f"\n{'='*100}\n  Step 1 — L34#{TARGET_NEURON} 当前 W_down 列分析\n{'='*100}", flush=True)
    W_down = layers[TARGET_LAYER].mlp.down_proj.weight.data
    v_old = W_down[:, TARGET_NEURON].detach().clone()
    print(f"  v_old norm = {v_old.float().norm().item():.4f}", flush=True)

    # vocab 投影
    proj = W_un.cpu() @ v_old.float().cpu()   # [vocab]
    proj_meow = proj[MEOW_ID].item()
    proj_period = proj[PERIOD_ID].item()
    proj_meow_normalized = proj_meow / W_un[MEOW_ID].float().norm().item() ** 2 \
                            * W_un[MEOW_ID].float().norm().item()
    print(f"  v_old @ W_unembed[喵] = {proj_meow:.4f}", flush=True)
    print(f"  v_old @ W_unembed[。] = {proj_period:.4f}", flush=True)

    # top push / suppress tokens
    top_pos = proj.topk(15)
    top_neg = (-proj).topk(15)
    print(f"\n  W_down[:, {TARGET_NEURON}] vocab top push 15:", flush=True)
    for v, tid in zip(top_pos.values, top_pos.indices):
        s = repr(tok.decode([tid.item()], skip_special_tokens=False))[:18]
        marker = " ★ 喵" if tid.item() == MEOW_ID else (" ★ 。" if tid.item() == PERIOD_ID else "")
        print(f"    {s:<20} {v.item():+.4f}{marker}", flush=True)

    print(f"\n  W_down[:, {TARGET_NEURON}] vocab top suppress 15:", flush=True)
    for v, tid in zip(top_neg.values, top_neg.indices):
        s = repr(tok.decode([tid.item()], skip_special_tokens=False))[:18]
        marker = " ★ 喵" if tid.item() == MEOW_ID else (" ★ 。" if tid.item() == PERIOD_ID else "")
        print(f"    {s:<20} {-v.item():+.4f}{marker}", flush=True)

    # Step 2: β sweep gen
    print(f"\n\n{'='*100}\n  Step 2 — β sweep generate test\n{'='*100}", flush=True)

    def gen(prompt, max_new=400):
        msgs = [{"role": "user", "content": prompt}]
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
        return tok.decode(gen_ids, skip_special_tokens=True), gen_ids

    results = {}
    meow_dir = W_un[MEOW_ID].clone()   # [hidden]
    for beta in BETAS:
        if beta == 0.0:
            # baseline, 不动 W_down
            pass
        else:
            W_down[:, TARGET_NEURON] = (v_old.float() + beta * meow_dir.float()).to(v_old.dtype)

        print(f"\n{'='*60}\n  β = {beta}\n{'='*60}", flush=True)
        results[beta] = {}
        for name, p in PROBES:
            text, gen_ids = gen(p, max_new=300)
            n_meow = text.count("喵")
            results[beta][name] = {"text": text, "n_meow": n_meow,
                                   "gen_len": len(gen_ids)}
            marker = f"  [喵×{n_meow}]" if n_meow > 0 else ""
            print(f"\n[{name}] {p!r}{marker}", flush=True)
            print(f"  → {text[:300]}", flush=True)
        W_down[:, TARGET_NEURON] = v_old   # 还原

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step31_meow_boost.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 31: L34#1312 喵 boost ================\n\n")
        f.write(f"v_old norm = {v_old.float().norm().item():.4f}\n")
        f.write(f"v_old @ W_unembed[喵] = {proj_meow:.4f}\n")
        f.write(f"v_old @ W_unembed[。] = {proj_period:.4f}\n\n")
        f.write(f"\n=== W_down 列 top push 15 ===\n")
        for v, tid in zip(top_pos.values, top_pos.indices):
            s = repr(tok.decode([tid.item()], skip_special_tokens=False))
            f.write(f"  {s:<22} {v.item():+.4f}\n")
        f.write(f"\n=== W_down 列 top suppress 15 ===\n")
        for v, tid in zip(top_neg.values, top_neg.indices):
            s = repr(tok.decode([tid.item()], skip_special_tokens=False))
            f.write(f"  {s:<22} {-v.item():+.4f}\n")
        f.write(f"\n=== β sweep ===\n")
        for beta in BETAS:
            f.write(f"\n--- β = {beta} ---\n")
            for name, p in PROBES:
                r = results[beta][name]
                f.write(f"\n[{name}] {p!r}  [喵×{r['n_meow']}, len={r['gen_len']}]\n")
                f.write(f"  → {r['text']}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
