"""
Phase 18: 知识段 inject — test "knowledge neuron" hypothesis

User insight: 之前一直造功能性神经元 (抽象通用), 失败. 改造知识性神经元 (case-specific,
只记得 leetcode 233 这类题怎么解).

最 minimal 实现: token-level inject 一段 Gemma-readable 知识段进 cot, 让 model follow.
如果 work, 验证 "case-specific knowledge inject + model 续生成" 这条 path 成立.

Inject 位置: cut body_ids 在 "We look at the digit $D$ at position $i$." 之后, 在 model 即将
写错的 "If D > 1: Contribution is 0" 之前. inject 我的 Gemma-readable 完整 Case 2 + verify.

Knowledge inject text 用 Gemma cot 风格 (用它自己的 P/A/D/B variable + LaTeX dollar 包裹).
"""
import os, sys, json, time
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
GEN_IDS_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_gen_ids.pt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase18_knowledge_inject.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_GEN = 3000

# inject 知识段 — Gemma-readable, 用它 cot 已建立的 P/A/D/B framework
KNOWLEDGE_INJECT = """\
        Wait — let me think carefully about what "prefix equal to $A$" actually means here. The prefix being fixed to $A$ does NOT mean every digit of $X$ is fixed. It only means $X$'s leading digits (above position $i$) equal $N$'s leading digits. The digit at position $i$ is what we are counting — we want cases where $X[i] = 1$.

        So when prefix($X$) $= A$ and $X[i] = 1$, we have $X = A \\times 10^{i+1} + 1 \\times P + \\text{suffix}$, where $\\text{suffix} \\in [0, P-1]$. There are $P$ such candidate $X$ values. We need $X \\le N$.

        Comparing $X$'s tail $(1 \\times P + \\text{suffix})$ to $N$'s tail $(D \\times P + B)$:

        *   If $D > 1$: $X$'s tail $\\le P + (P-1) = 2P - 1 < 2P \\le D \\times P \\le N$'s tail. All $P$ candidates satisfy $X \\le N$. **Contribution is $P$**.
        *   If $D = 1$: $X$'s tail $= P + \\text{suffix}$, $N$'s tail $= P + B$. Need suffix $\\le B$. **Contribution is $B + 1$**.
        *   If $D = 0$: $X$'s tail $\\ge P > B = N$'s tail. **Contribution is $0$**.

        Verify $N = 13$:
        $i = 0, P = 1$: $A = 1, D = 3, B = 0$. Prefix contribution: $1$. $D > 1$ adds $P = 1$. Total: $2$.
        $i = 1, P = 10$: $A = 0, D = 1, B = 3$. Prefix contribution: $0$. $D = 1$ adds $B + 1 = 4$. Total: $4$.
        Grand total: $2 + 4 = 6$. ✓

5.  **C++ Implementation:**

"""


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
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
            def pre_hook(m, inputs):
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
    for h in hooks: h.remove()


def find_token_position_for_char(tok, body_ids, char_target):
    decoded = ""
    for i, tid in enumerate(body_ids):
        decoded += tok.decode([tid], skip_special_tokens=False)
        if len(decoded) >= char_target:
            return i
    return -1


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
    sel_clamp = []
    for (li, n, dv) in p2["top_diff"][:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": int(li), "index": int(n), "gain": gain})

    body_ids = torch.load(GEN_IDS_PATH, weights_only=True).tolist()
    full_decoded = tok.decode(body_ids, skip_special_tokens=False)

    # 找 inject 位置 — 在 "We look at the digit $D$ at position $i$." 之后
    marker = "We look at the digit $D$ at position $i$."
    idx = full_decoded.find(marker)
    if idx < 0:
        print(f"ERROR: marker not found: {marker!r}")
        # try shorter
        marker = "We look at the digit"
        idx = full_decoded.find(marker)
        if idx < 0:
            print("still not found")
            return
    cut_char_pos = idx + len(marker)
    print(f"\n[inject point] char {cut_char_pos}")
    print(f"  prefix tail 100: {full_decoded[cut_char_pos-100:cut_char_pos]!r}")
    print(f"  原 cot 后续 100 chars (会被替换): {full_decoded[cut_char_pos:cut_char_pos+200]!r}")

    cut_tok_idx = find_token_position_for_char(tok, body_ids, cut_char_pos)
    cut_body = body_ids[:cut_tok_idx + 1]
    print(f"  cut at body token {cut_tok_idx}")

    # Build prefix
    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = chat_input["input_ids"][0].tolist()
    # New prefix = prompt + cut_body + tokenized(knowledge_inject)
    inject_ids = tok.encode(KNOWLEDGE_INJECT, add_special_tokens=False)
    new_prefix = prompt_ids + cut_body + inject_ids
    print(f"\n[new prefix] prompt={len(prompt_ids)} + cut_body={len(cut_body)} + inject={len(inject_ids)} = {len(new_prefix)} tokens")
    print(f"\n[inject text head 300]:\n{KNOWLEDGE_INJECT[:300]}")
    print(f"\n[inject text tail 200]:\n{KNOWLEDGE_INJECT[-200:]}")

    # Generate
    input_ids_t = torch.tensor([new_prefix], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(input_ids_t)

    hooks = install_clamp_hooks(layers, sel_clamp)
    print(f"\n[gen] K={K} clamp + knowledge inject in cot, max_new={MAX_GEN}...")
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=input_ids_t, attention_mask=attn,
                max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=[eot],
            )
    finally:
        remove_hooks(hooks)
    print(f"  gen done in {time.time()-t0:.1f}s")
    gen_ids = out[0][len(new_prefix):].cpu().tolist()
    stopped = (gen_ids and gen_ids[-1] == eot)
    while gen_ids and gen_ids[-1] == eot:
        gen_ids = gen_ids[:-1]
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"  gen_len={len(gen_ids)} stopped={stopped}")

    # Signals check
    import re
    sig = {
        "if (D > 1) (with += P)": bool(re.search(r"if\s*\([Dd]\s*>\s*1\)[^}]*count\s*\+=\s*P", gen_text, re.DOTALL)),
        "else if (D > 1)":         bool(re.search(r"else\s+if\s*\([Dd]\s*>\s*1\)", gen_text)),
        "count += P":              bool(re.search(r"count\s*\+=\s*P", gen_text)),
        "while (P <= n)":          bool(re.search(r"while\s*\(\s*P\s*\*?=?\s*<=?\s*n\s*\)|while\s*\(\s*power_of_10\s*<=?\s*n\s*\)", gen_text)),
        "n /= 10 (Bug B)":         bool(re.search(r"\bn\s*/=\s*10", gen_text)),
        "temp_n":                  "temp_n" in gen_text,
        "P *= 10":                 bool(re.search(r"P\s*\*=\s*10|power_of_10\s*\*=\s*10", gen_text)),
        "class Solution":          "class Solution" in gen_text,
    }
    print(f"\n[signals]")
    for k, v in sig.items():
        print(f"  {k}: {v}")

    # Extract C++ code
    m = re.search(r'class Solution.*?\};', gen_text, re.DOTALL)
    if m:
        print(f"\n--- code ({len(m.group(0))} chars) ---")
        print(m.group(0))
    else:
        print(f"\n[no complete code] tail 800:")
        print(gen_text[-800:])

    # Save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 18: knowledge inject ================\n\n")
        f.write(f"cut at body token {cut_tok_idx} (char {cut_char_pos})\n")
        f.write(f"inject {len(inject_ids)} tokens of knowledge\n")
        f.write(f"gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(f"signals:\n")
        for k, v in sig.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n---- inject text ----\n{KNOWLEDGE_INJECT}\n")
        f.write(f"\n---- generated continuation ----\n{gen_text}\n")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
