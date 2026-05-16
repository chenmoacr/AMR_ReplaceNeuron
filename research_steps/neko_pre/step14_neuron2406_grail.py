"""
Step 14: L28#2406 圣杯实验 + 神经元本质探查

Part 1 (圣杯实验): clamp L28#2406 验证因果性
  baseline:                 logit_diff baseline
  clamp = 0:                完全关掉这个神经元
  clamp = +11.3 (反向):     强制激活方向翻转 (原 -11.3)
  clamp = mean(其他 prompt): 用控制平均值替换

Part 2a (cross-prompt): L28#2406 在不同 self-id prompt 上的激活
  替换模型名: Gemma / Llama / Claude / GPT-4 / ChatGPT / Qwen / 文心 / ...
  对照: 非 self-id prompt
  → 判定是 Gemma-specific 还是通用 self-id 路由神经元

Part 2b (vocab direction): W_down[:, 2406] 的 vocab 投影
  v = layers[28].mlp.down_proj.weight[:, 2406]   shape [1536]
  做 final_norm + lm_head → top-30 / bottom-30 token
  → 揭示这个神经元在 vocab 空间指向什么方向
"""
from __future__ import annotations
import os, sys
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

USER_PROMPT = "你是谁?"
GEMMA_PREFIX_TEXT = "我是 Gemma 4，一个由"
GEMMA_PREFIX_IDS  = [44889, 147224, 236743, 236812, 236900, 5095, 237852]
GOOGLE_ID = 6475
OPENAI_ID = 131846
APPLE_ID  = 9947
TARGET_LAYER = 28
TARGET_NEURON = 2406

# Cross-prompt 测试集（替换模型名 / 对照）
CROSS_PROMPTS = [
    # name, user_prompt, prefix_text_to_continue
    ("Gemma_4",   "你是谁?", "我是 Gemma 4，一个由"),
    ("Llama_3",   "你是谁?", "我是 Llama 3，一个由"),
    ("Claude_3",  "你是谁?", "我是 Claude 3，一个由"),
    ("GPT_4",     "你是谁?", "我是 GPT-4，一个由"),
    ("ChatGPT",   "你是谁?", "我是 ChatGPT，一个由"),
    ("Qwen",      "你是谁?", "我是 Qwen，一个由"),
    ("Mistral",   "你是谁?", "我是 Mistral，一个由"),
    ("文心一言",   "你是谁?", "我是 文心一言，一个由"),
    ("豆包",      "你是谁?", "我是 豆包，一个由"),
    ("DeepSeek",  "你是谁?", "我是 DeepSeek，一个由"),
    # 对照：相同结构但非 self-id
    ("iPhone",    "iPhone 是哪家公司开发的?", "iPhone 是由"),
    ("Windows",   "Windows 是哪家公司开发的?", "Windows 是由"),
    ("Tesla",     "特斯拉是哪家公司开发的?", "特斯拉是由"),
    # 完全不同语境
    ("Sky",       "天空是什么颜色?", "天空是蓝色的，因为"),
    ("Math",      "1+1 等于多少?", "1 加 1 等于"),
    ("Food",      "你喜欢什么?", "我喜欢吃"),
]


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
    final_norm = text_model.norm
    lm_head = model.lm_head
    n_layers = len(layers)

    # ============================================================
    # Part 2a: cross-prompt 激活模式
    # ============================================================
    print(f"\n{'='*100}\n  Part 2a: cross-prompt L{TARGET_LAYER}#{TARGET_NEURON} 激活模式\n{'='*100}",
          flush=True)
    print(f"{'name':<12} {'user_prompt':<25} {'prefix':<35} "
          f"{'act':>10} {'next_top1':<14} {'p(top1)':>8}", flush=True)

    cross_results = []
    for name, user, prefix_text in CROSS_PROMPTS:
        msgs = [{"role": "user", "content": user}]
        pre_enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        prefix_ids_part = tok.encode(prefix_text, add_special_tokens=False)
        full_seq = pre_enc["input_ids"][0].tolist() + prefix_ids_part
        last_pos = len(full_seq) - 1
        ids_t = torch.tensor([full_seq], dtype=torch.long, device=DEVICE)

        captured = {"act": None}
        def hook(m, inputs):
            captured["act"] = inputs[0][0, last_pos, TARGET_NEURON].detach().clone()

        h = layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            logits = out.logits[0, last_pos].float().cpu()
        finally:
            h.remove()

        act = captured["act"].float().item()
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(5)
        top1_id = top.indices[0].item()
        top1_str = repr(tok.decode([top1_id], skip_special_tokens=False))
        p_g = probs[GOOGLE_ID].item()
        p_o = probs[OPENAI_ID].item()
        p_a = probs[APPLE_ID].item()
        diff = (logits[GOOGLE_ID] - logits[OPENAI_ID]).item()
        cross_results.append({
            "name": name, "user": user, "prefix": prefix_text,
            "last_pos": last_pos,
            "act_2406": act,
            "top1_id": top1_id, "top1_str": top1_str,
            "p_top1": top.values[0].item(),
            "p_google": p_g, "p_openai": p_o, "p_apple": p_a,
            "logit_diff_GO": diff,
            "top5_ids": top.indices.tolist(),
            "top5_probs": top.values.tolist(),
        })
        print(f"  {name:<12} {user[:24]:<25} {prefix_text[:34]:<35} "
              f"{act:>+10.3f} {top1_str[:14]:<14} {top.values[0].item():>8.4f}",
              flush=True)

    # 计算控制组（非 Gemma + 非 self-id）的均值用于 clamp 实验
    nonself_acts = [r["act_2406"] for r in cross_results
                     if r["name"] in ("iPhone", "Windows", "Tesla", "Sky", "Math", "Food")]
    selfid_other_acts = [r["act_2406"] for r in cross_results
                          if r["name"] in ("Llama_3", "Claude_3", "GPT_4", "ChatGPT",
                                            "Qwen", "Mistral", "文心一言", "豆包", "DeepSeek")]
    mean_nonself = sum(nonself_acts) / max(len(nonself_acts), 1)
    mean_selfid_other = sum(selfid_other_acts) / max(len(selfid_other_acts), 1)
    gemma_act = next(r["act_2406"] for r in cross_results if r["name"] == "Gemma_4")
    print(f"\n  Gemma_4 activation:           {gemma_act:>+8.3f}", flush=True)
    print(f"  mean(其他 self-id) activation: {mean_selfid_other:>+8.3f}", flush=True)
    print(f"  mean(非 self-id) activation:   {mean_nonself:>+8.3f}", flush=True)

    # ============================================================
    # Part 1: 圣杯实验 — clamp L28#2406
    # ============================================================
    print(f"\n{'='*100}\n  Part 1: 圣杯 — clamp L{TARGET_LAYER}#{TARGET_NEURON}\n{'='*100}",
          flush=True)

    # baseline forward
    msgs_g = [{"role": "user", "content": USER_PROMPT}]
    pre_enc_g = tok.apply_chat_template(
        msgs_g, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    full_g = pre_enc_g["input_ids"][0].tolist() + GEMMA_PREFIX_IDS
    last_pos_g = len(full_g) - 1
    ids_g = torch.tensor([full_g], dtype=torch.long, device=DEVICE)

    def run_with_clamp(clamp_value):
        """clamp_value: float or None (None = baseline)"""
        if clamp_value is None:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_g, use_cache=False)
            return out.logits[0, last_pos_g].float().cpu()

        def hook(m, inputs):
            x = inputs[0].clone()
            x[0, last_pos_g, TARGET_NEURON] = clamp_value
            return (x,) + inputs[1:]
        h = layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_g, use_cache=False)
            return out.logits[0, last_pos_g].float().cpu()
        finally:
            h.remove()

    def report(name, logits, baseline_diff=None):
        probs = F.softmax(logits, dim=-1)
        top = probs.topk(5)
        diff = (logits[GOOGLE_ID] - logits[OPENAI_ID]).item()
        marker = ""
        if baseline_diff is not None:
            marker = f"  (Δdiff = {diff - baseline_diff:+.3f})"
        print(f"\n  --- {name}{marker} ---", flush=True)
        print(f"  logit(Google)={logits[GOOGLE_ID].item():>+8.3f}  "
              f"logit(OpenAI)={logits[OPENAI_ID].item():>+8.3f}  "
              f"logit(Apple)={logits[APPLE_ID].item():>+8.3f}", flush=True)
        print(f"  diff(G-O)={diff:>+8.3f}  "
              f"p(G)={probs[GOOGLE_ID].item():.4f}  "
              f"p(OpenAI)={probs[OPENAI_ID].item():.4f}", flush=True)
        print(f"  top-5:", flush=True)
        for i, (p, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            t_str = repr(tok.decode([tid], skip_special_tokens=False))
            print(f"    {i+1}. {t_str:<22} {p:>.4f}", flush=True)
        return diff

    # baseline
    bl = run_with_clamp(None)
    bl_diff = report("baseline (no clamp)", bl)

    # clamp = 0
    l0 = run_with_clamp(0.0)
    report("clamp L28#2406 = 0 (关掉)", l0, bl_diff)

    # clamp = mean of other self-id prompts
    l_mid = run_with_clamp(mean_selfid_other)
    report(f"clamp = {mean_selfid_other:+.2f} (其他 self-id 均值)", l_mid, bl_diff)

    # clamp = mean of non-self-id (basically suppress)
    l_off = run_with_clamp(mean_nonself)
    report(f"clamp = {mean_nonself:+.2f} (非 self-id 均值)", l_off, bl_diff)

    # clamp = +11.3 (反向)
    l_rev = run_with_clamp(+11.3)
    report("clamp = +11.3 (强制反向)", l_rev, bl_diff)

    # 也试 clamp = -1.0 (减弱激活到接近 zero level 的 negative)
    l_weak = run_with_clamp(-1.0)
    report("clamp = -1.0 (减弱激活)", l_weak, bl_diff)

    # ============================================================
    # Part 2b: W_down[:, 2406] 的 vocab 投影
    # ============================================================
    print(f"\n{'='*100}\n  Part 2b: W_down[:, {TARGET_NEURON}] 的 vocab 投影方向\n{'='*100}",
          flush=True)
    W = layers[TARGET_LAYER].mlp.down_proj.weight   # [hidden, intermediate]
    print(f"  W_down shape: {tuple(W.shape)}", flush=True)
    v = W[:, TARGET_NEURON].float()   # [hidden=1536]
    print(f"  v = W_down[:, {TARGET_NEURON}]  norm = {v.norm().item():.4f}", flush=True)

    # 直接 v @ W_unembed.T 看 vocab 投影
    W_un = lm_head.weight.float()   # [vocab, hidden]
    proj = W_un @ v   # [vocab]
    top_pos = proj.topk(30)
    top_neg = (-proj).topk(30)

    print(f"\n  --- top 30 token 被 W_down[:,2406] 推 (positive direction) ---", flush=True)
    for i, (val, tid) in enumerate(zip(top_pos.values.tolist(), top_pos.indices.tolist())):
        print(f"    {i+1:>2}. {repr(tok.decode([tid], skip_special_tokens=False)):<22} "
              f"proj={val:>+8.4f}", flush=True)

    print(f"\n  --- top 30 token 被 W_down[:,2406] 反对 (negative direction) ---", flush=True)
    for i, (val, tid) in enumerate(zip(top_neg.values.tolist(), top_neg.indices.tolist())):
        print(f"    {i+1:>2}. {repr(tok.decode([tid], skip_special_tokens=False)):<22} "
              f"proj={-val:>+8.4f}", flush=True)

    # 注意: 神经元在 Gemma prompt 上 activation = -11.3 (negative)
    # 实际 mlp_output_contribution = act * v = -11.3 * v
    # 所以实际推动 = 把 W_down 的 NEGATIVE 方向加到 residual
    # → top "negative direction" 才是 Gemma prompt 上实际被推的 token
    print(f"\n  注意: Gemma prompt 上 act = {gemma_act:+.2f} (negative)", flush=True)
    print(f"        实际加到 residual 的方向 = act × v = {gemma_act:+.2f} × v", flush=True)
    print(f"        → 实际推的 token = '反对方向' 的 top (negative direction)", flush=True)
    print(f"        → 实际反对的 token = 'positive direction' 的 top", flush=True)

    # 同时也用 final_norm 处理 v 的标准 logit lens
    print(f"\n  --- 通过 final_norm + lm_head 的 logit lens (act=1) ---", flush=True)
    with torch.no_grad():
        v_dev = v.to(DEVICE).to(torch.bfloat16)
        v_n = final_norm(v_dev.unsqueeze(0).unsqueeze(0))
        logits_v = lm_head(v_n).squeeze().float().cpu()
    top_ll = F.softmax(logits_v, dim=-1).topk(30)
    print(f"  (注意: 这种 logit lens 对单 W_down 列做 norm 在数学上有点hacky，仅供参考)",
          flush=True)
    for i, (p, tid) in enumerate(zip(top_ll.values.tolist(), top_ll.indices.tolist())):
        print(f"    {i+1:>2}. {repr(tok.decode([tid], skip_special_tokens=False)):<22} "
              f"prob={p:>.4f}", flush=True)

    # ============================================================
    # 保存
    # ============================================================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step14_neuron2406_grail.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 14: L28#2406 圣杯 + 探查 ================\n\n")

        f.write("=== Part 2a: cross-prompt L28#2406 激活模式 ===\n")
        f.write(f"{'name':<12} {'prefix':<35} {'act':>10}  {'top1_next':<14} {'p(top1)':>8}  "
                f"{'p(G)':>8} {'p(O)':>8} {'p(A)':>8}  {'logit_diff_GO':>13}\n")
        for r in cross_results:
            f.write(f"  {r['name']:<12} {r['prefix'][:34]:<35} {r['act_2406']:>+10.3f}  "
                    f"{r['top1_str'][:14]:<14} {r['p_top1']:>.4f}  "
                    f"{r['p_google']:>.4f} {r['p_openai']:>.4f} {r['p_apple']:>.4f}  "
                    f"{r['logit_diff_GO']:>+13.3f}\n")
        f.write(f"\n  Gemma_4 activation:           {gemma_act:>+8.3f}\n")
        f.write(f"  mean(其他 self-id) activation: {mean_selfid_other:>+8.3f}\n")
        f.write(f"  mean(非 self-id) activation:   {mean_nonself:>+8.3f}\n")

        f.write(f"\n\n=== Part 1: 圣杯 — clamp L28#2406 ===\n")
        for name, val in [
            ("baseline", None),
            ("clamp = 0", 0.0),
            (f"clamp = {mean_selfid_other:+.2f} (other self-id mean)", mean_selfid_other),
            (f"clamp = {mean_nonself:+.2f} (non-self-id mean)", mean_nonself),
            ("clamp = +11.3 (反向)", +11.3),
            ("clamp = -1.0 (减弱)", -1.0),
        ]:
            logits = run_with_clamp(val)
            probs = F.softmax(logits, dim=-1)
            top = probs.topk(5)
            diff = (logits[GOOGLE_ID] - logits[OPENAI_ID]).item()
            f.write(f"\n--- {name} ---\n")
            f.write(f"  logit(G)={logits[GOOGLE_ID]:+8.3f}  "
                    f"logit(O)={logits[OPENAI_ID]:+8.3f}  "
                    f"logit(A)={logits[APPLE_ID]:+8.3f}\n")
            f.write(f"  diff(G-O)={diff:+8.3f}  "
                    f"p(G)={probs[GOOGLE_ID]:.4f}  "
                    f"p(OpenAI)={probs[OPENAI_ID]:.4f}\n")
            f.write(f"  top-5:\n")
            for i, (p, tid) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
                t_str = repr(tok.decode([tid], skip_special_tokens=False))
                f.write(f"    {i+1}. {t_str:<22} {p:>.4f}\n")

        f.write(f"\n\n=== Part 2b: W_down[:, 2406] vocab 方向 ===\n")
        f.write(f"  v = W_down[:, 2406]  norm = {v.norm().item():.4f}\n")
        f.write(f"  Gemma prompt 上 act = {gemma_act:+.3f} (negative!)\n")
        f.write(f"  → 实际加到 residual 的方向 = act × v\n")
        f.write(f"  → 推的 token 是 'positive proj × negative act = negative effect' 中的 top\n\n")

        f.write(f"  Top 30 token 被 +v 推 (W_down[:, 2406] @ W_unembed[t]):\n")
        for i, (val, tid) in enumerate(zip(top_pos.values.tolist(), top_pos.indices.tolist())):
            f.write(f"    {i+1:>2}. {repr(tok.decode([tid], skip_special_tokens=False)):<22} "
                    f"proj={val:>+8.4f}\n")
        f.write(f"\n  Top 30 token 被 -v 推 (即 Gemma 上实际被推的 token):\n")
        for i, (val, tid) in enumerate(zip(top_neg.values.tolist(), top_neg.indices.tolist())):
            f.write(f"    {i+1:>2}. {repr(tok.decode([tid], skip_special_tokens=False)):<22} "
                    f"proj={-val:>+8.4f}\n")

    torch.save({
        "config": {"target_layer": TARGET_LAYER, "target_neuron": TARGET_NEURON,
                    "gemma_prefix": GEMMA_PREFIX_TEXT},
        "cross_results": cross_results,
        "gemma_act": gemma_act,
        "mean_selfid_other": mean_selfid_other,
        "mean_nonself": mean_nonself,
        "v_norm": v.norm().item(),
        "top_pos_proj": [(tid, val) for val, tid in
                          zip(top_pos.values.tolist(), top_pos.indices.tolist())],
        "top_neg_proj": [(tid, -val) for val, tid in
                          zip(top_neg.values.tolist(), top_neg.indices.tolist())],
    }, OUT_DIR / "step14_neuron2406_grail_raw.pt")
    print(f"\n[save] step14_neuron2406_grail.txt", flush=True)
    print(f"[save] step14_neuron2406_grail_raw.pt", flush=True)


if __name__ == "__main__":
    main()
