"""
Phase 5g: handcraft on dead neurons L33#6417 (Bug A) + L34#12188 (Bug B).

不动现有 polysemantic. 用 step32 猫娘那套 mech:
  W_gate[L][i, :] = scale_g × v_disc_n     (detection: hidden state pattern 在错位点)
  W_up[L][i, :]   = scale_u × v_pos_n
  W_down[L][:, i] = α × v_push_n           (push correct token over bug)

step 1: forward phase4a sequence with K=30 clamp, hook L33/L34 gate_proj input
step 2: find "Contribution is " (Bug A) 和 "// 准备下一位" (Bug B) 位置
step 3: v_disc 取那位 hidden state (1536-d)
step 4: handcraft 两个 dead neuron 三向量
step 5: free-gen leetcode 233 with K=30 + 2 handcrafts (no inject)
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
K = 30
ABS_CLAMP = 1.5
SCALE_G = 0.2
SCALE_U = 0.2
ALPHA_A = 3.0   # step32 猫娘 sweet spot 起步, 后续可调
ALPHA_B = 3.0
MAX_NEW = 8000

BUG_A = {"L": 33, "I": 6417, "tag": "Bug A (cot D>1 → P)"}
BUG_B = {"L": 34, "I": 12188, "tag": "Bug B (code 准备下一位 → P)"}


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

    # 关键 marker 位置 (Bug A 错位点 + Bug B 错位点)
    marker_A = "*   If $D > 1$: The digit at position $i$ cannot be 1 (since the prefix is fixed). Contribution is "
    pos_A_char = phase4a_text.find(marker_A) + len(marker_A)
    prefix_A_text = phase4a_text[:pos_A_char]   # 到 "Contribution is " 之后的所有 text

    marker_B = "n /= 10;"
    pos_B_char = phase4a_text.find(marker_B)
    prefix_B_text = phase4a_text[:pos_B_char]

    print(f"[Bug A marker char {pos_A_char}] tail: {prefix_A_text[-80:]!r}")
    print(f"[Bug B marker char {pos_B_char}] tail: {prefix_B_text[-80:]!r}")

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

    # K=30 clamp selections
    p2 = torch.load(PHASE2_PT, weights_only=False)
    top_diff = p2["top_diff"]
    sel_clamp = []
    for (li, n, dv) in top_diff[:K]:
        gain = (1.0 if dv > 0 else -1.0) * ABS_CLAMP
        sel_clamp.append({"layer": li, "index": n, "gain": gain})

    # build full sequence for capture
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()

    body_A_ids = tok.encode(prefix_A_text, add_special_tokens=False)
    body_B_ids = tok.encode(prefix_B_text, add_special_tokens=False)
    full_A = torch.tensor(prompt_ids + body_A_ids, dtype=torch.long).to(DEVICE)
    full_B = torch.tensor(prompt_ids + body_B_ids, dtype=torch.long).to(DEVICE)

    # capture gate_proj input (= 1536-d hidden state) at last position with K=30 clamp on
    def capture_gate_input(full_ids, target_L):
        cap = {"h": None}
        def pre_hook(m, inputs):
            cap["h"] = inputs[0][0, -1, :].to(torch.float32).cpu().clone()
        h = layers[target_L].mlp.gate_proj.register_forward_pre_hook(pre_hook)
        hooks_c = install_clamp_hooks(layers, sel_clamp)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                _ = model(input_ids=full_ids.unsqueeze(0), use_cache=False)
        finally:
            h.remove()
            remove_hooks(hooks_c)
        return cap["h"]

    print("\n[capture L33 gate_proj input at Bug A position]")
    h_A = capture_gate_input(full_A, BUG_A["L"])
    print(f"  ||h_A|| = {h_A.norm().item():.3f}")

    print("[capture L34 gate_proj input at Bug B position]")
    h_B = capture_gate_input(full_B, BUG_B["L"])
    print(f"  ||h_B|| = {h_B.norm().item():.3f}")

    # signatures
    v_disc_A_n = F.normalize(h_A, dim=0)
    v_disc_B_n = F.normalize(h_B, dim=0)

    # push directions (in 1536-d hidden space; same as lm_head row dim)
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    dollar_id = tok.encode("$", add_special_tokens=False)[0]
    n_id = tok.encode("n", add_special_tokens=False)[0]
    P_id = tok.encode("P", add_special_tokens=False)[0]

    push_A = (lm_head.weight[dollar_id] - lm_head.weight[zero_id]).to(torch.float32).cpu()
    push_A_n = F.normalize(push_A, dim=0)
    push_B = (lm_head.weight[P_id] - lm_head.weight[n_id]).to(torch.float32).cpu()
    push_B_n = F.normalize(push_B, dim=0)
    print(f"\n[push_A: '$' over '0' direction] ||push_A||={push_A.norm().item():.3f}")
    print(f"[push_B: 'P' over 'n' direction] ||push_B||={push_B.norm().item():.3f}")

    # backup original dead neuron columns + rows
    backups = {}
    for B in (BUG_A, BUG_B):
        L, I = B["L"], B["I"]
        backups[(L, I, "gate")] = layers[L].mlp.gate_proj.weight.data[I, :].detach().clone()
        backups[(L, I, "up")] = layers[L].mlp.up_proj.weight.data[I, :].detach().clone()
        backups[(L, I, "down")] = layers[L].mlp.down_proj.weight.data[:, I].detach().clone()

    # handcraft
    def handcraft(L, I, v_disc_n, push_n, alpha):
        Wg = layers[L].mlp.gate_proj.weight.data
        Wu = layers[L].mlp.up_proj.weight.data
        Wd = layers[L].mlp.down_proj.weight.data
        v_disc_dev = v_disc_n.to(Wg.device, dtype=Wg.dtype)
        push_dev = push_n.to(Wd.device, dtype=Wd.dtype)
        Wg[I, :] = SCALE_G * v_disc_dev
        Wu[I, :] = SCALE_U * v_disc_dev   # use disc for both gate and up (simpler)
        Wd[:, I] = alpha * push_dev

    def restore_all():
        for B in (BUG_A, BUG_B):
            L, I = B["L"], B["I"]
            layers[L].mlp.gate_proj.weight.data[I, :] = backups[(L, I, "gate")]
            layers[L].mlp.up_proj.weight.data[I, :] = backups[(L, I, "up")]
            layers[L].mlp.down_proj.weight.data[:, I] = backups[(L, I, "down")]

    handcraft(BUG_A["L"], BUG_A["I"], v_disc_A_n, push_A_n, ALPHA_A)
    handcraft(BUG_B["L"], BUG_B["I"], v_disc_B_n, push_B_n, ALPHA_B)
    print(f"\n[handcraft applied]")
    print(f"  Bug A: L{BUG_A['L']}#{BUG_A['I']}  gate/up scale={SCALE_G}/{SCALE_U}  down α={ALPHA_A}")
    print(f"  Bug B: L{BUG_B['L']}#{BUG_B['I']}  gate/up scale={SCALE_G}/{SCALE_U}  down α={ALPHA_B}")

    # free gen
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    print(f"\n[gen] K={K} clamp + 2 handcrafted neurons, max_new={MAX_NEW}")
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
        print(f"  restored")
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
    code_present = bool(m)
    print(f"\n  code complete: {code_present}")
    if m:
        print(f"\n--- code block ({len(m.group(0))} chars) ---")
        print(m.group(0))

    save_path = OUT_DIR / "phase5g_handcraft.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5g handcraft ================\n\n")
        f.write(f"BugA: L33#6417, scale_g={SCALE_G}, scale_u={SCALE_U}, alpha={ALPHA_A}\n")
        f.write(f"BugB: L34#12188, scale_g={SCALE_G}, scale_u={SCALE_U}, alpha={ALPHA_B}\n")
        f.write(f"K={K} clamp + handcraft, max_new={MAX_NEW}\n")
        f.write(f"gen_len={len(gen_ids)} stopped={stopped}\n\n")
        f.write(f"signals:\n")
        for k, v in sig.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n---- gen text ----\n{text}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
