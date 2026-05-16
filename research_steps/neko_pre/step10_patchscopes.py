"""
Step 10: Patchscopes (修补镜) — 让模型自己"读"hidden state

来自 Patchscopes 论文 (Geva et al. 2024).

方法：
  1. Source forward: 跑 "我是 Gemma 4，一个由" 抓某层 last_pos 的 hidden state h_src
  2. Target prompt: "请用一两个词概括这是什么概念："
  3. 在 target 序列后追加一个 placeholder token
  4. Hook on layers[L_tgt] forward_pre_hook:
       在 prefill (预填充) 时把 placeholder 位置的 hidden state 替换为 h_src
  5. 让模型 generate (生成)，看输出
  → 模型生成的文本 = 它对 h_src 的"自省解读"

测试不同 layer 的 source hidden：
  L1 / L5 / L10 / L15 / L20 / L25 / L30 / L34
  期望：早层 hidden 解读模糊或噪声；深层 hidden 解读应该接近 "Google" / "AI" / "开发者"
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = ROOT / "identity_swap"
DEVICE = "cuda:0"

SOURCE_PROMPT_TEXT = "你是谁?"
SOURCE_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]   # 我是 Gemma 4，一个由

# 要扫描的 source layer
SOURCE_LAYERS = [0, 1, 3, 5, 8, 12, 16, 20, 24, 25, 27, 30, 34]

# Target prompts (要让模型解读 source hidden 是什么)
TARGET_TEMPLATES = {
    "中文_单词描述": "请用一个词概括这是什么：",
    "中文_概念问句": "这指的是什么？",
    "中文_关于谁": "请说说这是关于什么或谁的：",
    "英文_oneword":  "Describe in one word what this is about: ",
    "few_shot_中文": "苹果→水果\n汽车→交通工具\n书本→物品\n???→",
    "few_shot_英文": "apple→fruit\ncar→vehicle\nbook→item\n???→",
}

MAX_NEW_TOKENS = 25


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
    print(f"[load] {MODEL_PATH}", flush=True)
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
    text_model = model.model.language_model
    layers = text_model.layers
    n_layers = len(layers)
    eos_ids = get_eos_ids(tok)

    # ---- 1. Source forward, capture all hidden states ----
    msgs = [{"role": "user", "content": SOURCE_PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    src_prefix_ids = pre_enc["input_ids"][0].tolist()
    src_seq = src_prefix_ids + SOURCE_PREFIX_IDS
    src_last_pos = len(src_seq) - 1
    print(f"[source] T={len(src_seq)}  last_pos={src_last_pos}", flush=True)
    print(f"  decode: {tok.decode(src_seq)!r}", flush=True)

    captured_hiddens = {}    # {L: hidden tensor [1536]}
    handles = []
    for L in range(n_layers):
        def make_hook(L=L):
            def hook(m, i, o):
                x = o[0] if isinstance(o, tuple) else o
                captured_hiddens[L] = x[0, src_last_pos].detach().clone()
            return hook
        handles.append(layers[L].register_forward_hook(make_hook()))

    src_ids_t = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=src_ids_t, use_cache=False)
    finally:
        for h in handles: h.remove()

    print(f"  captured hidden states for L0..L{n_layers-1}  shape={tuple(captured_hiddens[0].shape)}",
          flush=True)
    for L in [0, 1, 5, 25, 34]:
        h = captured_hiddens[L].float()
        print(f"    L{L}: norm={h.norm().item():.2f}  mean={h.mean().item():+.4f}  std={h.std().item():.4f}",
              flush=True)

    # ---- 2. Patchscope helper ----
    def patchscope_generate(h_src, target_template, L_tgt, max_new=MAX_NEW_TOKENS):
        """
        h_src: [1536] tensor on device (source hidden state)
        target_template: str (target prompt text)
        L_tgt: int (which layer to patch into; usually = L_src)
        Returns: generated text string
        """
        # tokenize target
        tgt_msgs = [{"role": "user", "content": target_template}]
        tgt_enc = tok.apply_chat_template(
            tgt_msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        tgt_ids = tgt_enc["input_ids"][0].tolist()
        # 在末尾追加一个 placeholder token (将被 patch)
        # 用 pad_token_id 或 unk_token_id
        placeholder_id = tok.pad_token_id if tok.pad_token_id else 0
        tgt_ids_full = tgt_ids + [placeholder_id]
        patch_pos = len(tgt_ids)   # placeholder 的位置
        T = len(tgt_ids_full)

        ids_t = torch.tensor([tgt_ids_full], dtype=torch.long, device=DEVICE)
        attn = torch.ones_like(ids_t)

        # Hook on layer L_tgt: 在 prefill 时替换 placeholder 位置的 hidden state
        h_src_bf16 = h_src.to(torch.bfloat16)
        replaced = {"done": False}

        def patch_hook(m, i):
            x = i[0]
            # 仅在 prefill (输入序列长度 ≥ T) 触发
            if x.shape[1] >= T and not replaced["done"]:
                x = x.clone()
                x[0, patch_pos, :] = h_src_bf16
                replaced["done"] = True
                return (x,) + i[1:]
            return None   # 不修改

        h = layers[L_tgt].register_forward_pre_hook(patch_hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model.generate(
                    input_ids=ids_t, attention_mask=attn,
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=eos_ids if eos_ids else None,
                    use_cache=True,
                )
        finally:
            h.remove()

        gen = out[0, T:].cpu().tolist()
        while gen and gen[-1] in eos_ids:
            gen = gen[:-1]
        return tok.decode(gen, skip_special_tokens=True), replaced["done"]

    # ---- 3. Run patchscopes for each source layer × each target template ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []   # list of dicts
    print(f"\n{'='*100}", flush=True)
    print(f"  Patchscopes — let model interpret hidden state from source layer L_src", flush=True)
    print(f"{'='*100}", flush=True)

    # baseline: 不 patch，直接用 target template 让模型答（看 unprompted answer）
    print(f"\n--- Baseline: target prompt 不替换 hidden 时模型的回答 ---", flush=True)
    for tname, ttext in TARGET_TEMPLATES.items():
        msgs = [{"role": "user", "content": ttext}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
        gen = out[0, enc["input_ids"].shape[1]:].cpu().tolist()
        while gen and gen[-1] in eos_ids:
            gen = gen[:-1]
        baseline_text = tok.decode(gen, skip_special_tokens=True)
        print(f"  [{tname}] {ttext!r}", flush=True)
        print(f"    baseline → {baseline_text!r}", flush=True)

    # 主实验
    for tname, ttext in TARGET_TEMPLATES.items():
        print(f"\n{'='*100}\n  Target template: [{tname}] {ttext!r}\n{'='*100}",
              flush=True)
        for L_src in SOURCE_LAYERS:
            h = captured_hiddens[L_src].to(DEVICE)
            # 把 source hidden patch 到 target 的同一层 (L_tgt = L_src)
            text, ok = patchscope_generate(h, ttext, L_tgt=L_src)
            print(f"  L_src={L_src:>2}  → {text!r}  (patched={ok})", flush=True)
            results.append({
                "target_name": tname,
                "target_text": ttext,
                "L_src": L_src,
                "L_tgt": L_src,
                "h_norm": h.float().norm().item(),
                "gen_text": text,
                "patched": ok,
            })

    # ---- 4. save ----
    with open(OUT_DIR / "step10_patchscopes.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 10: Patchscopes ================\n\n")
        f.write(f"Source seq:  {tok.decode(src_seq)!r}\n")
        f.write(f"Source last_pos: {src_last_pos}\n\n")

        for tname, ttext in TARGET_TEMPLATES.items():
            f.write(f"\n{'='*70}\n  Target [{tname}]: {ttext!r}\n{'='*70}\n")
            for r in [r for r in results if r["target_name"] == tname]:
                f.write(f"  L_src={r['L_src']:>2}  h_norm={r['h_norm']:>7.2f}  "
                        f"→ {r['gen_text']!r}\n")

        f.write(f"\n\n{'='*70}\n  By source layer (across all targets)\n{'='*70}\n")
        for L_src in SOURCE_LAYERS:
            f.write(f"\n--- L_src = {L_src} ---\n")
            for r in [r for r in results if r["L_src"] == L_src]:
                f.write(f"  [{r['target_name']:<14}] {r['gen_text']!r}\n")

    torch.save({
        "config": {"src_seq": src_seq, "src_last_pos": src_last_pos,
                    "source_layers": SOURCE_LAYERS,
                    "target_templates": TARGET_TEMPLATES},
        "results": results,
    }, OUT_DIR / "step10_patchscopes_raw.pt")
    print(f"\n[save] step10_patchscopes.txt", flush=True)
    print(f"[save] step10_patchscopes_raw.pt", flush=True)


if __name__ == "__main__":
    main()
