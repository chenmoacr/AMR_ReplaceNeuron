"""
Phase 4c: 两件事
(A) 加倍抑制 L34#9243 (推 n/=10 bug 最大贡献), K=30 base + L34#9243 gain=-4.0 free-gen
    目标: code 选 P *= 10 而不是 n /= 10
(B) 在 D == 1 case close `}` 之后, attribution 找推 "skip D>1" 的最大神经元
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
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
EXTRA_NEG_L34_9243 = -2.5   # 在 K=30 基础上额外 push (总 gain = -1.5 + -2.5 = -4.0)
MAX_NEW = 6000
L_LO = 15


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

    # Find D == 1 close } position in phase4a text — for attribution prefix
    d1_match = re.search(r"if \(D == 1\) \{\s*\n\s*total_count \+= B \+ 1;\s*\n\s*\}", phase4a_text)
    if not d1_match:
        raise RuntimeError("can't find D==1 block")
    d1_end = d1_match.end()
    prefix_at_d1_close = phase4a_text[:d1_end]
    print(f"[attribution prefix] ends at char {d1_end}")
    print(f"  tail 100: {prefix_at_d1_close[-100:]!r}")

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
    n_layers = len(layers)
    lm_head = model.lm_head
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel_base = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_base.append({"layer": li, "index": n, "gain": gain})

    # ============================================================
    # (A) free-gen with K=30 base + L34#9243 extra negative
    # ============================================================
    print(f"\n{'='*60}\n(A) free-gen K=30 + L34#9243 extra gain {EXTRA_NEG_L34_9243}\n{'='*60}")

    # find L34#9243 in sel_base and add extra negative
    sel_a = [dict(s) for s in sel_base]
    found = False
    for s in sel_a:
        if s["layer"] == 34 and s["index"] == 9243:
            s["gain"] += EXTRA_NEG_L34_9243
            print(f"  L34#9243 gain: {s['gain']:+.2f}")
            found = True
            break
    if not found:
        sel_a.append({"layer": 34, "index": 9243, "gain": EXTRA_NEG_L34_9243})
        print(f"  L34#9243 added with gain {EXTRA_NEG_L34_9243:+.2f}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    hooks = install_clamp_hooks(layers, sel_a)
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
    text_a = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n[gen-A] len={len(gen_ids)} stopped={stopped}")
    # extract code
    m = re.search(r'class Solution.*?\};', text_a, re.DOTALL)
    if m:
        print(f"\n--- (A) code block ({len(m.group(0))} chars) ---")
        print(m.group(0))
    else:
        print(f"\n(A) NO complete code; tail 1500: {text_a[-1500:]}")

    # ============================================================
    # (B) Attribution at D == 1 close } position
    # ============================================================
    print(f"\n{'='*60}\n(B) attribution at D==1 close (predict next token: skip-D>1 vs continue)\n{'='*60}")

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(prefix_at_d1_close, add_special_tokens=False)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long)
    print(f"[full_ids] {full_ids.shape[0]}")

    # capture activations
    acts = [None] * n_layers
    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            acts[idx] = x[0, -1, :].to(torch.float32).cpu().clone()
        return fn
    hooks = install_clamp_hooks(layers, sel_base)
    cap_hooks = [layers[i].mlp.down_proj.register_forward_hook(make_hook(i))
                 for i in range(n_layers)]
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model(input_ids=full_ids.unsqueeze(0).to(DEVICE), use_cache=False)
    finally:
        remove_hooks(hooks)
        remove_hooks(cap_hooks)
    logits_last = out.logits[0, -1, :].to(torch.float32).cpu()

    # top-20 next tokens
    top = torch.topk(logits_last, 20)
    print(f"\n[top-20 next-token logit at D==1 close `}}`]")
    for r, (v, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        print(f"  {r+1:>2}  id={idx:<6d}  logit={v:+.3f}  {tok.decode([idx], skip_special_tokens=False)!r}")

    # candidate token analysis
    # "} else if" continuation: next token after `}` close of D==1 -> ` else` (with space) or `else`
    # "skip" continuation: next token -> `\n` (newline)
    candidates = {
        " else": tok.encode(" else", add_special_tokens=False),
        "else":  tok.encode("else", add_special_tokens=False),
        "\n":    tok.encode("\n", add_special_tokens=False),
        "\n\n":  tok.encode("\n\n", add_special_tokens=False),
        " //":   tok.encode(" //", add_special_tokens=False),
    }
    print(f"\n[candidate logit]")
    for name, ids in candidates.items():
        if ids:
            print(f"  {name!r:8s} -> id={ids[0]:<6d} logit={logits_last[ids[0]].item():+.3f}")

    # Attribution: which neurons push 'skip' (newline) over 'else if'?
    # diff direction = lm_head['\n'] - lm_head[' else']
    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    else_id = tok.encode(" else", add_special_tokens=False)[0]
    e_nl = lm_head.weight[nl_id].to(torch.float32).cpu()
    e_else = lm_head.weight[else_id].to(torch.float32).cpu()
    e_diff = e_nl - e_else  # positive contrib = push skip over D>1

    pairs = []
    for li in range(L_LO, n_layers):
        W = layers[li].mlp.down_proj.weight.data.to(torch.float32).cpu()
        proj = e_diff @ W
        contrib = acts[li] * proj
        for i in range(contrib.shape[0]):
            cv = contrib[i].item()
            pairs.append((abs(cv), li, i, cv, acts[li][i].item(), proj[i].item()))
    pairs.sort(reverse=True)
    print(f"\n=== top 30 neurons pushing '\\n' (skip D>1) over ' else' (write D>1) ===")
    print(f"  {'rank':>4} {'layer#nrn':>11} {'contrib':>9} {'act':>7} {'proj':>9}")
    for k, (_, li, n, cv, av, pv) in enumerate(pairs[:30]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {cv:+.4f}  {av:+.3f}  {pv:+.5f}")

    # save
    save_path = OUT_DIR / "phase4c.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("================ Phase 4c ================\n\n")
        f.write(f"(A) K=30 base + L34#9243 extra gain {EXTRA_NEG_L34_9243} free-gen\n")
        f.write(f"    gen_len={len(gen_ids)} stopped={stopped}\n")
        if m:
            f.write(f"    code: {m.group(0)}\n")
        else:
            f.write(f"    no code block; tail: {text_a[-1500:]}\n")
        f.write(f"\n\n(B) attribution at D==1 close `}}` (predicts what?)\n")
        f.write(f"    candidate logits:\n")
        for name, ids in candidates.items():
            if ids:
                f.write(f"      {name!r:8s} -> logit={logits_last[ids[0]].item():+.3f}\n")
        f.write(f"\n    top 30 push 'skip' over 'else':\n")
        for k, (_, li, n, cv, av, pv) in enumerate(pairs[:30]):
            f.write(f"      L{li:02d}#{n:<5d}  contrib={cv:+.4f}  act={av:+.3f}  proj={pv:+.5f}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
