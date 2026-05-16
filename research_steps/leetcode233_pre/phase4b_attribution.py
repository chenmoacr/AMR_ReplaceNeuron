"""
Phase 4b': 在 K=30 clamp 下 teacher-force prefix 到 "// 准备下一位\n        " 这个位置,
单次 forward 抓最后一位 hidden state + logit, 算每个 neuron 对 "n" vs "P" 的贡献.

两个 prefix:
- P0: 原 phase4a code 直到 "// 准备下一位\n        " 前 (含 D == 1 但漏 D > 1)
- P1: 同 P0 但插入了 D>1 修复

对每个 prefix:
1. 看 top-K next-token logit (model 想写什么)
2. logit_diff = logit("n") - logit("P") -- 正值 = 偏 bug, 负值 = 偏对
3. 每个 layer 每个 neuron 的贡献 = act[L,i] * (W_down[L][:, i] @ (e_n - e_P))
4. top |contrib| neurons = 推 bug 的最大候选 (反向 clamp 目标)
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
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4b_attribution.txt"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
TOP_K_NEURONS = 30
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


def extract_phase4a_text(text):
    """phase4a header 之后是 raw generated text."""
    nl = text.find("\n\n")
    if nl < 0:
        return text
    return text[nl+2:]


def split_at_inject_point(phase4a_text):
    """找 phase4a text 中 'n /= 10;' 之前的 "// 准备下一位\n        " 位置.
    Returns (prefix_text_before_inject_target, inject_target_word, suffix)
    where inject_target_word should be 'n' (bug) or 'P' (correct)."""
    # 找到 "n /= 10;" 第一次出现
    bug_marker = "n /= 10;"
    idx = phase4a_text.find(bug_marker)
    if idx < 0:
        raise RuntimeError("can't find 'n /= 10;' in phase4a output")
    # prefix = 一切到 "n /= 10;" 的 "n" 之前
    return phase4a_text[:idx], "n", phase4a_text[idx+1:]


def build_two_prefixes(phase4a_text):
    """P0 = 原样到 inject target ('n' 之前)
    P1 = 原样直到 "// 准备下一位" 之前, 插入 D>1 修复, 然后接 "// 准备下一位" 到 'n' 之前
    """
    P0_prefix, target, _ = split_at_inject_point(phase4a_text)

    prep_marker = "// 准备下一位"
    pidx = P0_prefix.rfind(prep_marker)
    if pidx < 0:
        raise RuntimeError("can't find '// 准备下一位'")
    # 找到 D == 1 case 结束的位置 (右括号 } 然后空行)
    d1_close = P0_prefix.rfind("}", 0, pidx)
    if d1_close < 0:
        raise RuntimeError("can't find D==1 closing }")
    # inject 在 d1_close 之后, prep_marker 之前
    indent = "        "
    inject = " else if (D > 1) {\n" + indent + "    total_count += P;\n" + indent + "}"
    P1_prefix = P0_prefix[:d1_close+1] + inject + P0_prefix[d1_close+1:]

    return P0_prefix, P1_prefix, target


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = extract_phase4a_text(phase4a_full)

    P0_text, P1_text, target_word = build_two_prefixes(phase4a_text)
    print(f"[prefix] target='{target_word}' next-token")
    print(f"[P0 len] {len(P0_text)} chars (original, no fix)")
    print(f"[P1 len] {len(P1_text)} chars (with D>1 fix injected)")
    # show last 200 chars of each
    print(f"\n[P0 tail]\n{P0_text[-200:]!r}")
    print(f"\n[P1 tail]\n{P1_text[-200:]!r}")

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
    sel = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel.append({"layer": li, "index": n, "gain": gain})

    # tokenize candidate words "n" and "P" (need to match how they appear after spaces)
    # In phase4a text, after "// 准备下一位\n        " the next char is 'n' (or 'P' if fixed)
    # Tokenize " n" "  n" "n" etc. and see what model would emit
    # Actually since prefix ends with the spaces, the model emits 'n' or 'P' or "P" or other as standalone token

    def tok_candidate(word):
        ids = tok.encode(word, add_special_tokens=False)
        return ids

    n_ids = tok_candidate("n")
    P_ids = tok_candidate("P")
    print(f"\n[tok] 'n' -> {n_ids}  decode->{tok.decode(n_ids)!r}")
    print(f"[tok] 'P' -> {P_ids}  decode->{tok.decode(P_ids)!r}")

    # build full ids for each prefix
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()

    # phase4a text already includes everything from "<|channel>thought\nThinking Process:..."
    # to "</channel|>response..." etc.
    # but our P0/P1 start from this position so just append prompt_ids + tokenize(P0/P1)
    # Actually phase4a_text starts with "<|channel>thought\n" already so just direct encode

    def encode_full(prefix_text):
        body_ids = tok.encode(prefix_text, add_special_tokens=False)
        full = list(prompt_ids) + body_ids
        return torch.tensor(full, dtype=torch.long)

    P0_ids = encode_full(P0_text)
    P1_ids = encode_full(P1_text)
    print(f"\n[P0 token len] {P0_ids.shape[0]}")
    print(f"[P1 token len] {P1_ids.shape[0]}")

    # capture activations + final hidden at last position
    def forward_with_capture(full_ids):
        T = full_ids.shape[0]
        acts = [None] * n_layers   # [in_features] at last position
        def make_hook(idx):
            def fn(module, inputs, output):
                x = inputs[0]
                acts[idx] = x[0, -1, :].to(torch.float32).cpu().clone()
            return fn
        hooks = install_clamp_hooks(layers, sel)
        cap_hooks = [layers[i].mlp.down_proj.register_forward_hook(make_hook(i))
                     for i in range(n_layers)]
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=full_ids.unsqueeze(0).to(DEVICE),
                            use_cache=False)
        finally:
            remove_hooks(hooks)
            remove_hooks(cap_hooks)
        logits_last = out.logits[0, -1, :].to(torch.float32).cpu()
        return logits_last, acts

    print("\n[fwd P0]...")
    logits_P0, acts_P0 = forward_with_capture(P0_ids)
    print("[fwd P1]...")
    logits_P1, acts_P1 = forward_with_capture(P1_ids)

    def report_logits(tag, logits):
        top = torch.topk(logits, 20)
        print(f"\n[{tag} top-20 next-token logit]")
        for r, (v, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            decoded = tok.decode([idx], skip_special_tokens=False)
            print(f"  {r+1:>2}  id={idx:<6d}  logit={v:+.3f}  {decoded!r}")
        # specific compare
        n_id = n_ids[0] if n_ids else None
        P_id = P_ids[0] if P_ids else None
        if n_id is not None and P_id is not None:
            print(f"  logit('n') = {logits[n_id].item():+.3f}")
            print(f"  logit('P') = {logits[P_id].item():+.3f}")
            print(f"  diff = {(logits[n_id] - logits[P_id]).item():+.3f}  (positive = bug-leaning)")

    report_logits("P0 (no fix)", logits_P0)
    report_logits("P1 (D>1 fixed)", logits_P1)

    # Attribution: per neuron contribution to (logit('n') - logit('P'))
    # contrib[L][i] = act[L][i] * (W_down[L][:, i] @ (lm_head.weight[n_id] - lm_head.weight[P_id]))
    n_id = n_ids[0]
    P_id = P_ids[0]
    e_n = lm_head.weight[n_id].to(torch.float32).cpu()
    e_P = lm_head.weight[P_id].to(torch.float32).cpu()
    e_diff = e_n - e_P   # direction in residual stream that pushes 'n' over 'P'

    def attribute(acts):
        per_layer_neuron = []
        for li in range(n_layers):
            if li < L_LO:
                continue
            W_down = layers[li].mlp.down_proj.weight.data.to(torch.float32).cpu()  # [hidden, intermediate]
            # W_down[:, i] @ e_diff -> scalar per neuron
            proj = e_diff @ W_down   # [intermediate]
            contrib = acts[li] * proj  # element-wise
            for i in range(contrib.shape[0]):
                cv = contrib[i].item()
                per_layer_neuron.append((abs(cv), li, i, cv, acts[li][i].item(), proj[i].item()))
        per_layer_neuron.sort(reverse=True)
        return per_layer_neuron

    print("\n[attribution P0]...")
    attr_P0 = attribute(acts_P0)
    print("[attribution P1]...")
    attr_P1 = attribute(acts_P1)

    def report_attr(tag, attr_pairs, top=TOP_K_NEURONS):
        print(f"\n=== {tag}: top {top} neurons pushing 'n' over 'P' ===")
        print(f"  {'rank':>4} {'layer#nrn':>11} {'contrib':>9} {'act':>7} {'proj':>9}")
        for k, (_, li, n, cv, av, pv) in enumerate(attr_pairs[:top]):
            print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {cv:+.4f}  {av:+.3f}  {pv:+.5f}")

    report_attr("P0 (no fix)", attr_P0)
    report_attr("P1 (D>1 fixed)", attr_P1)

    # Save
    save_data = {
        "qa_id": "phase4b_attribution",
        "n_layers": n_layers,
        "K_clamp": K,
        "ABS_CLAMP": ABS_CLAMP,
        "P0_token_len": P0_ids.shape[0],
        "P1_token_len": P1_ids.shape[0],
        "n_id": n_id, "P_id": P_id,
        "logits_P0_top20": [(int(i), float(v)) for v, i in zip(*torch.topk(logits_P0, 20))],
        "logits_P1_top20": [(int(i), float(v)) for v, i in zip(*torch.topk(logits_P1, 20))],
        "diff_P0": (logits_P0[n_id] - logits_P0[P_id]).item(),
        "diff_P1": (logits_P1[n_id] - logits_P1[P_id]).item(),
        "attr_P0_top": [(li, n, cv) for (_, li, n, cv, _, _) in attr_P0[:50]],
        "attr_P1_top": [(li, n, cv) for (_, li, n, cv, _, _) in attr_P1[:50]],
    }
    torch.save(save_data, OUT_PATH.with_suffix(".pt"))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Phase 4b' attribution: 'n /= 10' vs 'P *= 10' at first bug position\n")
        f.write(f"K_clamp={K} ABS_CLAMP={ABS_CLAMP}\n\n")
        f.write(f"--- P0 (no fix) tail 200 chars ---\n{P0_text[-200:]!r}\n\n")
        f.write(f"--- P1 (D>1 fix) tail 200 chars ---\n{P1_text[-200:]!r}\n\n")
        f.write(f"\n=== logit summary ===\n")
        f.write(f"P0: logit('n')={logits_P0[n_id].item():+.3f}  logit('P')={logits_P0[P_id].item():+.3f}  "
                f"diff={(logits_P0[n_id]-logits_P0[P_id]).item():+.3f}\n")
        f.write(f"P1: logit('n')={logits_P1[n_id].item():+.3f}  logit('P')={logits_P1[P_id].item():+.3f}  "
                f"diff={(logits_P1[n_id]-logits_P1[P_id]).item():+.3f}\n")
        for tag, attr in [("P0", attr_P0), ("P1", attr_P1)]:
            f.write(f"\n=== {tag} top 50 neurons pushing 'n' over 'P' ===\n")
            for k, (_, li, n, cv, av, pv) in enumerate(attr[:50]):
                f.write(f"  L{li:02d}#{n:<5d}  contrib={cv:+.4f}  act={av:+.3f}  proj={pv:+.5f}\n")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
