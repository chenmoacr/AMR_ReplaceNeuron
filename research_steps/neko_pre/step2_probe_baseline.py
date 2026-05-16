"""
Step 2: 用 4 个 self-introduction prompt 在贪婪解码下让 Gemma 4 E2B 自我介绍，
找出 'Google' / '谷歌' / 'google' / 'Gemma' 等关键 token 在每段回答里
的出现次数与位置。

目标：选出"Google 出现一次、上下文清晰"的 prompt 作为差分主战场。

Output: identity_swap/step2_probe_baseline.txt
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT = Path("J:/amr/amr_wtf/identity_swap/step2_probe_baseline.txt")
DEVICE = "cuda:0"
MAX_NEW = 200

PROMPTS = [
    ("zh1", "请介绍一下你自己"),
    ("zh2", "你是谁?"),
    ("en1", "Who are you?"),
    ("en2", "Tell me about yourself."),
]

# 关键 token id（来自 step1）
TARGETS = {
    "Google":   19540,
    " Google":  6475,
    "google":   7239,
    " google":  21752,
    "GOOGLE":   202927,
    "谷歌":      122438,
    "Gemma":    None,  # 待查
    " Gemma":   None,
    "Apple":    31319,
    " Apple":   9947,
    "苹果":      68004,
}


def get_eos_ids(tokenizer):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return list({i for i in eos if i is not None})


def main():
    print(f"[load] {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    # 补齐 Gemma id
    TARGETS["Gemma"]  = tok.encode("Gemma",  add_special_tokens=False)[0]
    TARGETS[" Gemma"] = tok.encode(" Gemma", add_special_tokens=False)[0]
    print(f"  Gemma id={TARGETS['Gemma']}   ' Gemma' id={TARGETS[' Gemma']}", flush=True)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    eos_ids = get_eos_ids(tok)

    lines = ["# Step 2: Gemma 4 E2B 自我介绍贪婪解码扫描\n"]
    results = []

    for tag, prompt in PROMPTS:
        print(f"\n{'='*70}\n  [{tag}] {prompt!r}\n{'='*70}", flush=True)
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        prompt_len = input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids if eos_ids else None,
                use_cache=True,
            )
        dt = time.time() - t0
        full = out[0].detach().cpu().tolist()
        gen = full[prompt_len:]
        # 去掉尾部 eos
        while gen and gen[-1] in eos_ids:
            gen = gen[:-1]
        text = tok.decode(gen, skip_special_tokens=False)
        text_clean = tok.decode(gen, skip_special_tokens=True)

        # 找 target token 出现位置
        hits = {}
        for name, tid in TARGETS.items():
            if tid is None: continue
            positions = [i for i, t in enumerate(gen) if t == tid]
            if positions:
                hits[name] = positions

        print(f"  prompt_tokens={prompt_len}  gen_tokens={len(gen)}  dt={dt:.1f}s", flush=True)
        print(f"  --- TEXT (skip_special=True) ---\n{text_clean}", flush=True)
        print(f"  --- token-hits ---", flush=True)
        for name, ps in hits.items():
            print(f"    {name!r:>12}  count={len(ps)}  positions={ps}", flush=True)

        results.append({
            "tag": tag, "prompt": prompt,
            "prompt_len": prompt_len, "gen_len": len(gen),
            "gen_ids": gen, "text": text, "text_clean": text_clean,
            "hits": hits,
        })

    # ---- 总结 ----
    lines.append("\n## 摘要\n")
    lines.append(f"{'tag':<5} {'prompt_len':>10} {'gen_len':>7}  hits\n")
    lines.append("-" * 100 + "\n")
    for r in results:
        h = ", ".join(f"{k}×{len(v)}" for k, v in r["hits"].items())
        lines.append(f"{r['tag']:<5} {r['prompt_len']:>10} {r['gen_len']:>7}  {h}\n")

    lines.append("\n## 完整回答\n")
    for r in results:
        lines.append(f"\n--- [{r['tag']}] prompt = {r['prompt']!r} ---\n")
        lines.append(f"prompt_tokens={r['prompt_len']}  gen_tokens={r['gen_len']}\n")
        lines.append(f"hits={r['hits']}\n")
        lines.append(f"text:\n{r['text_clean']}\n")
        lines.append(f"--- raw with special:\n{r['text']}\n")

    lines.append("\n## token-by-token (gen 部分前 80 tokens)\n")
    for r in results:
        lines.append(f"\n--- [{r['tag']}] ---\n")
        for i, tid in enumerate(r["gen_ids"][:80]):
            piece = repr(tok.decode([tid], skip_special_tokens=False))
            lines.append(f"  {i:>3}  {tid:>7}  {piece}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
