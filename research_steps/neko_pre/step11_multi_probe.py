"""
Step 11: 多面 Patchscope (修补镜) 探查

问题：同一个 L25 hidden state，除了 'Google' 之外还含有什么信息？
方法：用 12+ 种不同的 probe (探针) prompt 让模型从不同角度"释放"hidden 里的信息。

source: "我是 Gemma 4，一个由" 在 L25 last_pos 的 hidden state h_25

probe (探针) 类型：
  A. 单点概括   (one-word / 一个词)
  B. 列表枚举   (列出 N 个)
  C. 详细描述   (开放式描述)
  D. 排除性     ("除了 X 还有什么")
  E. 角色扮演   (内心独白 / 心理活动)
  F. 分类       (属于什么领域)
  G. 场景       (这是什么场景)
  H. few-shot   (类比模板)
  I. 元问题     ("这是个 AI 的内部状态，描述它")

输出：
  step11_multi_probe.txt
  step11_multi_probe_raw.pt
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
SOURCE_PREFIX_IDS = [44889, 147224, 236743, 236812, 236900, 5095, 237852]
SOURCE_LAYER = 25     # step 9/10 已确认 L25 hidden 含 Google 信号最强
MAX_NEW_TOKENS = 120

PROBES = [
    # (probe_name, prompt_text)
    ("A1_单词",        "用一个词概括这是什么："),
    ("A2_oneword",     "Describe this in one word: "),

    ("B1_五个相关词",  "请列出 5 个与这个相关的关键词，用逗号分隔："),
    ("B2_十个keywords","List 10 keywords associated with this, comma-separated: "),

    ("C1_详细描述",    "请详细描述这是什么、关于什么、涉及什么。请尽可能多地说出你能感知到的所有信息："),
    ("C2_describe_all","Describe everything you can perceive about this. Tell me what it relates to, what it concerns, all aspects: "),

    ("D1_除了Google",  "这个东西的内容除了和 Google 有关，还涉及什么？请尽量列出："),
    ("D2_other",       "Besides being related to Google, what other concepts does this contain? List as many as possible: "),

    ("E1_内心独白",    "如果这是一个 AI 在某一时刻的内心独白，请详细描述它在想什么："),
    ("E2_role",        "Imagine this is the inner state of an AI assistant. Describe what it is thinking about in detail: "),

    ("F1_分类",        "这个内容属于哪些领域、类别或主题？请尽可能多地分类："),

    ("G1_场景",        "如果这是一个场景的内部表征，请描述这个场景的人物、对话、内容："),

    ("H1_few_shot_zh", "向量A → 动物、宠物、可爱、毛茸茸、喵\n向量B → 汽车、四轮、发动机、速度感、马路\n向量C → "),
    ("H2_few_shot_en", "Vector A → animal, pet, cute, fluffy, meow\nVector B → car, four wheels, engine, speed, road\nVector C → "),

    ("I1_meta",        "这是一个大语言模型在某一刻的内部表征。请详细分析它包含哪些信息片段、哪些主题、哪些情境："),

    ("J1_直问",        "请描述这里包含的所有信息，包括但不限于：人物、公司、产品、概念、情绪、场景。越详细越好："),
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
    eos_ids = get_eos_ids(tok)

    # ---- source forward, capture L25 hidden ----
    msgs = [{"role": "user", "content": SOURCE_PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    src_prefix = pre_enc["input_ids"][0].tolist()
    src_seq = src_prefix + SOURCE_PREFIX_IDS
    src_last = len(src_seq) - 1
    print(f"[source] T={len(src_seq)}  last_pos={src_last}  decode={tok.decode(src_seq)!r}",
          flush=True)

    captured = {}
    def hook(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        captured["h"] = x[0, src_last].detach().clone()
    h = layers[SOURCE_LAYER].register_forward_hook(hook)
    src_t = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=src_t, use_cache=False)
    finally:
        h.remove()
    h_src = captured["h"]
    print(f"[L{SOURCE_LAYER}] hidden norm={h_src.float().norm().item():.2f}",
          flush=True)

    # ---- patchscope helper ----
    def patchscope(probe_text, max_new=MAX_NEW_TOKENS):
        msgs_t = [{"role": "user", "content": probe_text}]
        enc = tok.apply_chat_template(
            msgs_t, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        tgt_ids = enc["input_ids"][0].tolist()
        placeholder = tok.pad_token_id if tok.pad_token_id else 0
        tgt_full = tgt_ids + [placeholder]
        patch_pos = len(tgt_ids)
        T = len(tgt_full)

        ids_t = torch.tensor([tgt_full], dtype=torch.long, device=DEVICE)
        attn = torch.ones_like(ids_t)
        h_bf = h_src.to(torch.bfloat16)
        replaced = {"done": False}

        def patch_pre(m, i):
            x = i[0]
            if x.shape[1] >= T and not replaced["done"]:
                x = x.clone()
                x[0, patch_pos, :] = h_bf
                replaced["done"] = True
                return (x,) + i[1:]
            return None

        h_handle = layers[SOURCE_LAYER].register_forward_pre_hook(patch_pre)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model.generate(
                    input_ids=ids_t, attention_mask=attn,
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=eos_ids,
                    use_cache=True,
                )
        finally:
            h_handle.remove()

        gen = out[0, T:].cpu().tolist()
        while gen and gen[-1] in eos_ids:
            gen = gen[:-1]
        return tok.decode(gen, skip_special_tokens=True), replaced["done"]

    # ---- baseline (no patch) for each probe ----
    def baseline(probe_text, max_new=MAX_NEW_TOKENS):
        msgs_t = [{"role": "user", "content": probe_text}]
        enc = tok.apply_chat_template(
            msgs_t, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
        gen = out[0, enc["input_ids"].shape[1]:].cpu().tolist()
        while gen and gen[-1] in eos_ids:
            gen = gen[:-1]
        return tok.decode(gen, skip_special_tokens=True)

    # ---- run all probes ----
    print(f"\n{'='*100}", flush=True)
    print(f"  Running {len(PROBES)} probes on L{SOURCE_LAYER} hidden", flush=True)
    print(f"{'='*100}", flush=True)
    results = []
    for pname, ptext in PROBES:
        print(f"\n--- {pname}: {ptext[:60]!r} ---", flush=True)

        bl = baseline(ptext)
        print(f"  [BASELINE]  {bl[:200]!r}", flush=True)

        ps, ok = patchscope(ptext)
        print(f"  [PATCHED]   {ps[:300]!r}  (patched={ok})", flush=True)

        results.append({"name": pname, "prompt": ptext,
                         "baseline": bl, "patched": ps, "ok": ok})

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step11_multi_probe.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 11: 多面 Patchscope ================\n\n")
        f.write(f"Source: {tok.decode(src_seq)!r}\n")
        f.write(f"Source layer: L{SOURCE_LAYER} at last_pos\n")
        f.write(f"hidden norm: {h_src.float().norm().item():.4f}\n\n")

        for r in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"  [{r['name']}]\n")
            f.write(f"  prompt: {r['prompt']!r}\n")
            f.write(f"{'='*70}\n")
            f.write(f"\n  -- BASELINE (no patch) --\n")
            f.write(f"  {r['baseline']}\n")
            f.write(f"\n  -- PATCHED (with L{SOURCE_LAYER} hidden) --\n")
            f.write(f"  {r['patched']}\n")

        # 统计：patched 输出里 'Google' 出现次数 & 其他实体
        f.write(f"\n\n{'='*70}\n  Keyword statistics across all patched outputs\n{'='*70}\n")
        keywords = {
            "Google":   0, "DeepMind": 0, "Gemma":    0, "Apple":    0,
            "OpenAI":   0, "GPT":      0, "AI":       0, "model":    0,
            "Microsoft":0, "助手":     0, "语言":     0, "训练":     0,
            "open":     0, "权重":     0, "weights":  0, "公司":     0,
            "开发":     0, "developer":0, "Bard":     0, "Gemini":   0,
        }
        for r in results:
            for k in keywords:
                keywords[k] += r["patched"].count(k)
        f.write(f"keyword counts (sum across {len(results)} patched outputs):\n")
        for k, c in sorted(keywords.items(), key=lambda x: -x[1]):
            f.write(f"  {k:<12}  {c}\n")

    torch.save({
        "config": {"src_seq": src_seq, "src_last": src_last,
                    "source_layer": SOURCE_LAYER},
        "h_src": h_src.float().cpu(),
        "results": results,
    }, OUT_DIR / "step11_multi_probe_raw.pt")
    print(f"\n[save] step11_multi_probe.txt", flush=True)
    print(f"[save] step11_multi_probe_raw.pt", flush=True)


if __name__ == "__main__":
    main()
