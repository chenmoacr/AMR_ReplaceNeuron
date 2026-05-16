"""
Phase 5h: handcraft with multi-position signature (per Opus single-sample 限制 mitigation).

5g 失败: single-position signature 太 narrow, gen 中 stutter.
5h 改: forward 完整 phase4a sequence, 找所有 "Contribution is " mentions, 抓 mean signature.

不只 Bug A. Bug B 同样 multi-position 处理 (phase4a code 段 'n /= 10' 只 1 处, 退化到 single-pos; 不期望太好).
"""
import os, sys, json, re
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
PHASE2_PT = ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
CHUNK = 1024
K = 30
ABS_CLAMP = 1.5
SCALE_G = 0.2
SCALE_U = 0.2
ALPHA_A = 3.0
ALPHA_B = 3.0
MAX_NEW = 8000

BUG_A = {"L": 33, "I": 6417}
BUG_B = {"L": 34, "I": 12188}


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


def find_token_positions_for_marker(tokenizer, full_ids, prompt_len, marker_str):
    """find all positions in full_ids where the token preceding `marker_str` start is at."""
    # decode whole assistant body
    body_ids = full_ids[prompt_len:].tolist()
    decoded = tokenizer.decode(body_ids, skip_special_tokens=False)
    positions = []
    start = 0
    while True:
        idx = decoded.find(marker_str, start)
        if idx < 0:
            break
        # binary search: smallest tok index ti such that decode(body_ids[:ti+1]) length >= idx (we want position OF the last token before marker emits)
        # i.e. token whose hidden state predicts the first char of marker
        lo, hi = 0, len(body_ids)
        target = idx
        while lo < hi:
            mid = (lo + hi) // 2
            t = tokenizer.decode(body_ids[: mid + 1], skip_special_tokens=False)
            if len(t) > target:
                hi = mid
            else:
                lo = mid + 1
        # lo is the token whose decode includes char `idx`. The last token BEFORE marker emission is lo-1.
        # But signature抓 model 即将选下一个 token 那位 = position whose hidden predicts the next token (= the marker's first token).
        # If marker spans tokens [k, k+m), then position k-1's hidden -> predicts token k.
        # We want hidden at token k-1.
        absolute_pos = prompt_len + lo - 1
        if absolute_pos >= prompt_len:
            positions.append(absolute_pos)
        start = idx + len(marker_str)
    return positions


def forward_capture_at_positions(model, layers, target_L, full_ids, positions, sel_clamp):
    """forward full_ids with K=30 clamp, hook target_L mlp.gate_proj input, capture hidden at given positions."""
    captured = []
    chunk_offset = {"v": 0}
    def pre_hook(m, inputs):
        x = inputs[0]
        chunk_len = x.shape[1]
        gs = chunk_offset["v"]
        for p in positions:
            if gs <= p < gs + chunk_len:
                captured.append((p, x[0, p - gs, :].to(torch.float32).cpu().clone()))
    h = layers[target_L].mlp.gate_proj.register_forward_pre_hook(pre_hook)
    hooks_c = install_clamp_hooks(layers, sel_clamp)
    past_kv = None
    pos = 0
    T = full_ids.shape[0]
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            while pos < T:
                end = min(pos + CHUNK, T)
                chunk = full_ids[pos:end].unsqueeze(0).to(DEVICE)
                chunk_offset["v"] = pos
                cache_pos = torch.arange(pos, end, device=DEVICE)
                out = model(input_ids=chunk, past_key_values=past_kv,
                            use_cache=True, cache_position=cache_pos)
                past_kv = out.past_key_values
                pos = end
                torch.cuda.empty_cache()
    finally:
        h.remove()
        remove_hooks(hooks_c)
    return captured


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = phase4a_full[phase4a_full.find("\n\n")+2:]

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
    lm_head = model.lm_head
    eot = tok.convert_tokens_to_ids("<turn|>")

    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel_clamp = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": li, "index": n, "gain": gain})

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(phase4a_text, add_special_tokens=False)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long)
    print(f"[full_ids] {full_ids.shape[0]} prompt_len={len(prompt_ids)}")

    # locate Bug A positions (all "Contribution is " in cot)
    pos_A = find_token_positions_for_marker(tok, full_ids, len(prompt_ids), "Contribution is ")
    print(f"\n[Bug A positions ({len(pos_A)}) found in cot for 'Contribution is ']: {pos_A[:10]}...")
    # locate Bug B positions (only "n /= 10" in code, expect 1)
    pos_B = find_token_positions_for_marker(tok, full_ids, len(prompt_ids), "n /= 10")
    print(f"[Bug B positions ({len(pos_B)}) found for 'n /= 10']: {pos_B}")

    print(f"\n[capture L33 at Bug A positions]")
    cap_A = forward_capture_at_positions(model, layers, BUG_A["L"], full_ids, pos_A, sel_clamp)
    h_A_stack = torch.stack([h for _, h in cap_A])
    h_A_mean = h_A_stack.mean(0)
    print(f"  captured {len(cap_A)} positions, ||mean||={h_A_mean.norm().item():.3f}")

    if pos_B:
        print(f"[capture L34 at Bug B positions]")
        cap_B = forward_capture_at_positions(model, layers, BUG_B["L"], full_ids, pos_B, sel_clamp)
        h_B_stack = torch.stack([h for _, h in cap_B])
        h_B_mean = h_B_stack.mean(0)
        print(f"  captured {len(cap_B)} positions, ||mean||={h_B_mean.norm().item():.3f}")
    else:
        h_B_mean = None

    # signatures normalized
    v_disc_A_n = F.normalize(h_A_mean, dim=0)
    if h_B_mean is not None:
        v_disc_B_n = F.normalize(h_B_mean, dim=0)
    else:
        v_disc_B_n = None

    # push directions
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    dollar_id = tok.encode("$", add_special_tokens=False)[0]
    n_id = tok.encode("n", add_special_tokens=False)[0]
    P_id = tok.encode("P", add_special_tokens=False)[0]
    push_A_n = F.normalize((lm_head.weight[dollar_id] - lm_head.weight[zero_id]).to(torch.float32).cpu(), dim=0)
    push_B_n = F.normalize((lm_head.weight[P_id] - lm_head.weight[n_id]).to(torch.float32).cpu(), dim=0)

    # backups
    backups = {}
    for B in (BUG_A, BUG_B):
        L, I = B["L"], B["I"]
        backups[(L, I, "gate")] = layers[L].mlp.gate_proj.weight.data[I, :].detach().clone()
        backups[(L, I, "up")]   = layers[L].mlp.up_proj.weight.data[I, :].detach().clone()
        backups[(L, I, "down")] = layers[L].mlp.down_proj.weight.data[:, I].detach().clone()

    # handcraft
    def hc(L, I, v_disc_n, push_n, alpha):
        Wg = layers[L].mlp.gate_proj.weight.data
        Wu = layers[L].mlp.up_proj.weight.data
        Wd = layers[L].mlp.down_proj.weight.data
        v = v_disc_n.to(Wg.device, dtype=Wg.dtype)
        p = push_n.to(Wd.device, dtype=Wd.dtype)
        Wg[I, :] = SCALE_G * v
        Wu[I, :] = SCALE_U * v
        Wd[:, I] = alpha * p

    def restore_all():
        for B in (BUG_A, BUG_B):
            L, I = B["L"], B["I"]
            layers[L].mlp.gate_proj.weight.data[I, :] = backups[(L, I, "gate")]
            layers[L].mlp.up_proj.weight.data[I, :]   = backups[(L, I, "up")]
            layers[L].mlp.down_proj.weight.data[:, I] = backups[(L, I, "down")]

    hc(BUG_A["L"], BUG_A["I"], v_disc_A_n, push_A_n, ALPHA_A)
    if v_disc_B_n is not None:
        hc(BUG_B["L"], BUG_B["I"], v_disc_B_n, push_B_n, ALPHA_B)
    print(f"\n[handcraft applied with multi-pos signature]")
    print(f"  Bug A: L{BUG_A['L']}#{BUG_A['I']}  ({len(pos_A)} positions mean)  α={ALPHA_A}")
    if v_disc_B_n is not None:
        print(f"  Bug B: L{BUG_B['L']}#{BUG_B['I']}  ({len(pos_B)} positions mean)  α={ALPHA_B}")

    # free gen
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    print(f"\n[gen] K={K} clamp + 2 handcraft multi-pos, max_new={MAX_NEW}")
    hooks = install_clamp_hooks(layers, sel_clamp)
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
        restore_all()
    gen_ids = out[0][input_ids.shape[1]:].cpu().tolist()
    stopped = (gen_ids and gen_ids[-1] == eot)
    while gen_ids and gen_ids[-1] == eot:
        gen_ids = gen_ids[:-1]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"\n[gen] len={len(gen_ids)} stopped={stopped}")

    sig = {
        "Contribution is 0": len(re.findall(r"Contribution is 0", text)),
        "Contribution is $P$": len(re.findall(r"Contribution is \$P\$|Contribution is \$P", text)),
        "if (D > 1)": bool(re.search(r"if\s*\(\s*[Dd]\s*>\s*1\s*\)", text)),
        "count += P": bool(re.search(r"count\s*\+=\s*P\b", text)),
        "temp_n": "temp_n" in text,
        "n /= 10": bool(re.search(r"\bn\s*/=\s*10", text)),
        "power_of_10 *= 10": bool(re.search(r"power_of_10\s*\*=\s*10", text)),
        "string DP regression": bool(re.search(r"string repr|S\[i\]|positions \$i", text)),
        "stutter (any 5+ repeat)": bool(re.search(r"(.)\1{4,}", text)),
    }
    for k, v in sig.items():
        print(f"  {k}: {v}")

    m = re.search(r'class Solution.*?\};', text, re.DOTALL)
    print(f"\n  code complete: {bool(m)}")
    if m:
        print(f"\n--- code ({len(m.group(0))} chars) ---")
        print(m.group(0))

    save_path = OUT_DIR / "phase5h_multi_pos.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5h: multi-position signature handcraft ================\n\n")
        f.write(f"BugA: L33#6417, n_pos={len(pos_A)}, α={ALPHA_A}\n")
        f.write(f"BugB: L34#12188, n_pos={len(pos_B)}, α={ALPHA_B}\n")
        f.write(f"gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(f"signals:\n")
        for k, v in sig.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n---- gen ----\n{text}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
