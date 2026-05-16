"""
Phase 4c (revised): attribution at cot 段第一次 D>1 推导错的位置.

prefix 直到 cot 中 "If $D > 1$: ... Contribution is " 之后那位.
model 决定下一个 token: "0" (bug, K=30 clamp 实际选的) vs "$P$" / "P" (correct)

抓 last position hidden state per layer + 算每 neuron 对 logit('0') - logit('$') 的贡献.
top |contrib| = 推 cot 段错推导 "贡献 0" 的最大神经元 = 反向 clamp 目标 (cot 段, 不是 code 段).

跟 Opus 的诊断一致: framing 神经元 (phase 2 抓的) clamp 改了, 但 "推导步骤神经元" (这次抓的) 没改.
这次找的就是后者.
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
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
K = 30
ABS_CLAMP = 1.5
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

    # 找到 cot 第一次 D>1 错位点
    marker = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    idx = phase4a_text.find(marker)
    if idx < 0:
        raise RuntimeError(f"marker not found")
    prefix_end = idx + len(marker)
    prefix_text = phase4a_text[:prefix_end]
    print(f"[prefix end char] {prefix_end}")
    print(f"[prefix tail 200] {prefix_text[-200:]!r}")
    print(f"[next 30 chars in original] {phase4a_text[prefix_end:prefix_end+30]!r}")

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

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel.append({"layer": li, "index": n, "gain": gain})

    # tokenize candidate tokens
    cands = {
        "0":    tok.encode("0", add_special_tokens=False),
        "$":    tok.encode("$", add_special_tokens=False),
        "P":    tok.encode("P", add_special_tokens=False),
        "$P$":  tok.encode("$P$", add_special_tokens=False),
        " $P$": tok.encode(" $P$", add_special_tokens=False),
        " 0":   tok.encode(" 0", add_special_tokens=False),
        "$0":   tok.encode("$0", add_special_tokens=False),
        "P$":   tok.encode("P$", add_special_tokens=False),
    }
    print(f"\n[candidates]")
    for name, ids in cands.items():
        print(f"  {name!r:8s} -> {ids}  decode={tok.decode(ids)!r}")

    # build full sequence
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(prefix_text, add_special_tokens=False)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long)
    print(f"\n[full_ids] {full_ids.shape[0]}")

    # forward with K=30 clamp + capture activations at last position
    acts = [None] * n_layers
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
            out = model(input_ids=full_ids.unsqueeze(0).to(DEVICE), use_cache=False)
    finally:
        remove_hooks(hooks)
        remove_hooks(cap_hooks)
    logits_last = out.logits[0, -1, :].to(torch.float32).cpu()

    # top 20 next tokens
    top = torch.topk(logits_last, 20)
    print(f"\n[top-20 next-token logit at 'Contribution is ' position]")
    for r, (v, idx2) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        print(f"  {r+1:>2}  id={idx2:<6d}  logit={v:+.3f}  {tok.decode([idx2], skip_special_tokens=False)!r}")

    # candidate logits
    print(f"\n[candidate logits]")
    for name, ids in cands.items():
        if ids:
            print(f"  {name!r:8s} -> id={ids[0]:<6d} logit={logits_last[ids[0]].item():+.3f}")

    # Choose 'bug' vs 'correct' candidate
    # bug = "0" (single token, K=30 actually emitted)
    # correct = "$P$" (likely follows the prior pattern "Contribution is $...$")
    # We'll attribute on diff = lm_head[bug] - lm_head[correct]
    bug_id = cands["0"][0]
    # correct candidate: most likely "$" then "P" "$", so primary token is "$"
    # but more precise: "$P$" likely tokenizes as ["$", "P", "$"] — first token is "$"
    # bug "0" is also probably preceded by "$" actually... wait look at the context:
    # original was "Contribution is 0." (not "Contribution is $0$").
    # so model emitted "0" directly (no $). similarly correct would be "$P$" with leading "$"
    # so attribution direction = "0" vs "$"
    correct_id = cands["$"][0]
    print(f"\n[attribution direction] bug='{tok.decode([bug_id])!r}' (id={bug_id}) vs correct='{tok.decode([correct_id])!r}' (id={correct_id})")
    print(f"  logit('0') = {logits_last[bug_id].item():+.3f}")
    print(f"  logit('$') = {logits_last[correct_id].item():+.3f}")
    print(f"  diff = {(logits_last[bug_id]-logits_last[correct_id]).item():+.3f}  (>0=bug)")

    e_bug = lm_head.weight[bug_id].to(torch.float32).cpu()
    e_cor = lm_head.weight[correct_id].to(torch.float32).cpu()
    e_diff = e_bug - e_cor   # positive contrib = push bug

    pairs = []
    for li in range(L_LO, n_layers):
        W = layers[li].mlp.down_proj.weight.data.to(torch.float32).cpu()
        proj = e_diff @ W
        contrib = acts[li] * proj
        for i in range(contrib.shape[0]):
            cv = contrib[i].item()
            pairs.append((abs(cv), li, i, cv, acts[li][i].item(), proj[i].item()))
    pairs.sort(reverse=True)
    print(f"\n=== top 40 neurons pushing '0' (bug) over '$' (correct: $P$ start) at cot 'Contribution is ' ===")
    print(f"  {'rank':>4} {'layer#nrn':>11} {'contrib':>9} {'act':>7} {'proj':>9}")
    for k, (_, li, n, cv, av, pv) in enumerate(pairs[:40]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {cv:+.4f}  {av:+.3f}  {pv:+.5f}")

    # save
    save_path = OUT_DIR / "phase4c_cot_attribution.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("================ Phase 4c (revised, cot 段 attribution) ================\n\n")
        f.write(f"K_clamp={K} ABS_CLAMP={ABS_CLAMP}\n")
        f.write(f"prefix end at char {prefix_end} of phase4a text\n")
        f.write(f"prefix tail: {prefix_text[-200:]!r}\n\n")
        f.write(f"original next 30 chars: {phase4a_text[prefix_end:prefix_end+30]!r}\n\n")
        f.write(f"--- top 20 next-token logit ---\n")
        for r, (v, idx2) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            f.write(f"  id={idx2:<6d}  logit={v:+.3f}  {tok.decode([idx2], skip_special_tokens=False)!r}\n")
        f.write(f"\n--- candidate logits ---\n")
        for name, ids in cands.items():
            if ids:
                f.write(f"  {name!r:8s} -> logit={logits_last[ids[0]].item():+.3f}\n")
        f.write(f"\nbug='{tok.decode([bug_id])!r}' logit={logits_last[bug_id].item():+.3f}\n")
        f.write(f"correct='{tok.decode([correct_id])!r}' logit={logits_last[correct_id].item():+.3f}\n")
        f.write(f"diff (>0=bug): {(logits_last[bug_id]-logits_last[correct_id]).item():+.3f}\n\n")
        f.write(f"=== top 40 neurons pushing bug over correct ===\n")
        for k, (_, li, n, cv, av, pv) in enumerate(pairs[:40]):
            f.write(f"  L{li:02d}#{n:<5d}  contrib={cv:+.4f}  act={av:+.3f}  proj={pv:+.5f}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
