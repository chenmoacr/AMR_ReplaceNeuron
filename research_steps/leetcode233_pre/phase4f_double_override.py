"""
Phase 4f: 二次 token override 修 Bug B (loop 用 temp_n + n/=10).

prefix = phase4a (含 D>1 inject "$P$") + phase4e continuation 直到 'power_of_10 = 1;' 那行 + inject
inject = '        while (power_of_10 <= n) {\n            '  (替代 model 想写的 'long long temp_n = n;\n\n        while (temp_n > 0) {')
然后让 model 续 body, 看是否:
  - body 用 n 而非 temp_n
  - 不再写 n /= 10 或类似 step
  - trace n=13 → 6
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
PHASE4E_LOG = ROOT / "outputs" / "gemma_code_GB01" / "phase4e_run.log"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
MAX_NEW = 4000


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

    # Step 1: rebuild phase 4e prefix = phase4a cot 直到 D>1 markder + override
    marker = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    idx = phase4a_text.find(marker)
    if idx < 0:
        raise RuntimeError("D>1 marker not found")
    phase4e_prefix = phase4a_text[:idx + len(marker)] + "$P$."

    # Step 2: phase 4e continuation 直到 'long long power_of_10 = 1; // Represents 1, 10, 100, ... (P)\n'
    log_text = PHASE4E_LOG.read_text(encoding="utf-8", errors="replace")
    # find phase 4e generated continuation in the log
    pwr_marker = "long long power_of_10 = 1; // Represents 1, 10, 100, ... (P)"
    pwr_idx = log_text.find(pwr_marker)
    if pwr_idx < 0:
        raise RuntimeError("can't find power_of_10 marker in phase4e log")
    # we need everything from start of code in phase4e (look for "long long count = 0;" before pwr_idx)
    count_idx = log_text.rfind("long long count = 0;", 0, pwr_idx)
    if count_idx < 0:
        raise RuntimeError("can't find count = 0 marker")
    # snippet between these markers
    phase4e_code_head = log_text[count_idx:pwr_idx + len(pwr_marker)] + "\n        "
    print(f"[phase4e_code_head tail 300]:\n{phase4e_code_head[-300:]!r}")

    # Step 3: build full prefix text:
    #   phase4e_prefix + (continuation from after D>1 inject up to code_head)
    # Problem: I don't have the continuation between the D>1 inject point and "long long count = 0;"
    # I have the FULL continuation in phase4e log (from "[gen] len=..." or before the code block)
    # Let me reconstruct: phase4e_prefix ends with "$P$." then model gen continues.
    # In phase 4e the continuation contains all from after $P$ to end of code.
    # I need to find the continuation in log_text — the log has the prefix tail printed + then signals + gen tail
    # Actually the simpler approach: just reconstruct by re-decoding from the model's gen — but I'd have to rerun.
    # Alternative: in log_text find a unique marker right after $P$. and dump from there.

    # In phase4e the prefix_text_overridden ends with: "... Contribution is $P$."
    # then model generated text was decoded and printed (in [gen] continuation head 1500 chars).
    # easier: just dump the log section "--- continuation head 1500 chars ---" to "---" or to "code block 1"
    # actually let me re-read the log structure
    cont_head_marker = "--- continuation head 1500 chars ---\n"
    cont_idx = log_text.find(cont_head_marker)
    if cont_idx < 0:
        raise RuntimeError("can't find continuation head marker in log")
    # continuation starts after this
    cont_start = cont_idx + len(cont_head_marker)
    # ends at "--- code block" or "[signals" (whatever comes first)
    end1 = log_text.find("--- code block", cont_start)
    end2 = log_text.find("\n[signals", cont_start)
    end = min(e for e in [end1, end2, len(log_text)] if e > 0)
    cont_head_text = log_text[cont_start:end]
    # This is only 1500 chars - we need the full continuation between $P$. and 'long long count = 0;' to power_of_10
    # Hmm. Let's just use cont_head_text up to first 'long long count = 0;' if present
    cnt_in_head = cont_head_text.find("long long count = 0;")
    if cnt_in_head > 0:
        between = cont_head_text[:cnt_in_head]  # 从 $P$. 之后到 'long long count = 0;' 之前
        # full prefix = phase4e_prefix + between + phase4e_code_head 中包含 count = 0 起到 power_of_10
        full_prefix = phase4e_prefix + between + phase4e_code_head
    else:
        # head 1500 不含 count = 0, 说明 model 在中间还写了很多.
        # 简单: 用 between=cont_head_text + 后续 code (从 'long long count = 0;' 开始)
        full_prefix = phase4e_prefix + cont_head_text + "..." + phase4e_code_head
        print(f"[warn] 'long long count = 0;' not in 1500 head, using truncated assembly")

    # Now inject Bug B fix: 替代 'long long temp_n = n;\n\n        while (temp_n > 0) {'
    # 我们的 full_prefix 末尾是 '... power_of_10 = 1; ... (P)\n        '
    # 现在加 'while (power_of_10 <= n) {\n            ' (跳过 long long temp_n = n;)
    inject_b = "while (power_of_10 <= n) {\n            "
    full_prefix_with_inject = full_prefix + inject_b
    print(f"\n[full prefix with inject tail 400]:\n{full_prefix_with_inject[-400:]!r}")

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

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(full_prefix_with_inject, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids + body_ids], dtype=torch.long).to(DEVICE)
    attn = torch.ones_like(full_ids)
    print(f"\n[tokens] prompt={len(prompt_ids)}  body={len(body_ids)}  total={full_ids.shape[1]}")

    print(f"\n[gen] K={K} clamp, max_new={MAX_NEW}")
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
    cont_text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n[gen] len={len(gen_ids)} stopped={stopped}")

    # signals
    has_temp_n = "temp_n" in cont_text
    has_n_div = bool(re.search(r"\bn\s*/=\s*10", cont_text))
    has_p_mult = bool(re.search(r"power_of_10\s*\*=\s*10", cont_text))
    has_d_gt_1 = bool(re.search(r"if\s*\(\s*[Dd]\s*>\s*1\s*\)", cont_text))
    has_count_p = bool(re.search(r"count\s*\+=\s*P", cont_text))
    print(f"\n[signals in continuation]")
    print(f"  contains 'temp_n':        {has_temp_n}")
    print(f"  contains 'n /= 10':       {has_n_div}")
    print(f"  contains 'power_of_10 *=': {has_p_mult}")
    print(f"  has if (D > 1):           {has_d_gt_1}")
    print(f"  has count += P:           {has_count_p}")

    # find full code: combine inject + cont
    full_code_attempt = inject_b + cont_text
    m = re.search(r'class Solution.*?\};', cont_text, re.DOTALL)
    if not m:
        # try to extract from gen end: maybe gen continues from inside class body
        # in this case cont_text starts with body interior; look for closing } twice
        print(f"\n[continuation head 1500]:\n{cont_text[:1500]}")
        print(f"\n[continuation tail 1500]:\n{cont_text[-1500:]}")

    # save
    save_path = OUT_DIR / "phase4f_double_override.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 4f: double override (D>1 from 4e + while/loop) ================\n\n")
        f.write(f"K={K} ABS_CLAMP={ABS_CLAMP} max_new={MAX_NEW} gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(f"signals:\n")
        f.write(f"  has 'temp_n':           {has_temp_n}\n")
        f.write(f"  has 'n /= 10':          {has_n_div}\n")
        f.write(f"  has 'power_of_10 *=':   {has_p_mult}\n")
        f.write(f"  has if (D > 1):         {has_d_gt_1}\n")
        f.write(f"  has count += P:         {has_count_p}\n\n")
        f.write(f"---- inject ----\n{inject_b!r}\n\n")
        f.write(f"---- full prefix with inject (last 600 chars) ----\n{full_prefix_with_inject[-600:]}\n\n")
        f.write(f"---- continuation (full) ----\n{cont_text}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
