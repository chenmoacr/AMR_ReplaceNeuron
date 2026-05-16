"""
预计算猫娘神经元 signature, 存成 meow_signature.pt 供 UI 启动加载。
复用 step32 的逻辑: 从 step30_samples.json 抓 L33 mlp.gate_proj 输入,
分离 "句号前一位" (positive) vs 其他 assistant 位置 (negative)。
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
DEVICE = "cuda:0"
TARGET_L = 33
TARGET_I = 6417
# B 神经元 (喵抑制): L34 最 dead 的位置
B_LAYER = 34
B_INDEX = 12188

SAMPLES_JSON = Path(__file__).resolve().parents[2] / "research_steps" / "neko_pre" / "step30_samples.json"
OUT_PATH = Path(__file__).resolve().parent / "meow_signature.pt"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    MEOW_ID = tok.encode("喵", add_special_tokens=False)[0]
    PERIOD_ID = tok.encode("。", add_special_tokens=False)[0]
    print(f"[token] '喵'={MEOW_ID}, '。'={PERIOD_ID}", flush=True)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    layers = model.model.language_model.layers
    lm_head = model.lm_head

    samples_json = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    positive_h, negative_h = [], []
    # 用于 v_meow_context 的两路抓取 (L33 layer output)
    meow_out_h = []       # catmeowline 中 '喵 前一位' 的 L33 layer output
    period_out_h_base = []  # baseline 中 '。 前一位' 的 L33 layer output
    # 用于 v_just_meowed 的两路抓取 (L34 gate input)
    catmeow_just_meowed = []  # catmeow 中 ids[k-1]=喵 且 ids[k]=。 的 L34 gate input
    baseline_period_l34 = []   # baseline 中 ids[k]=。 的 L34 gate input

    def forward_and_capture(text, prompt, ids_out, gate_input_out, layer_out, l34_gate_out):
        msgs_full = [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": text}]
        enc = tok.apply_chat_template(
            msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        ids = enc["input_ids"][0].tolist()
        msgs_u = [{"role": "user", "content": prompt}]
        pre = tok.apply_chat_template(
            msgs_u, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        assistant_start = pre["input_ids"].shape[1]

        captured = {}
        def pre_hook(m, inputs):
            captured["gate_in"] = inputs[0][0].detach().float().cpu()
        def post_hook(m, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            captured["layer_out"] = out[0].detach().float().cpu()
        def pre_hook_l34(m, inputs):
            captured["gate_in_l34"] = inputs[0][0].detach().float().cpu()
        h1 = layers[TARGET_L].mlp.gate_proj.register_forward_pre_hook(pre_hook)
        h2 = layers[TARGET_L].register_forward_hook(post_hook)
        h3 = layers[B_LAYER].mlp.gate_proj.register_forward_pre_hook(pre_hook_l34)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                model(input_ids=enc["input_ids"].to(DEVICE), use_cache=False)
        finally:
            h1.remove()
            h2.remove()
            h3.remove()
        ids_out.append(ids)
        gate_input_out.append(captured["gate_in"])
        layer_out.append(captured["layer_out"])
        l34_gate_out.append(captured["gate_in_l34"])
        return ids, assistant_start

    for s_idx, s in enumerate(samples_json):
        prompt = s["input"]
        baseline = s["baseline"]
        catmeow = s.get("catmeowline")

        # === baseline pass: 抓 v_disc / v_pos signature + L33 layer output + L34 gate input ===
        b_ids_holder, b_gate_holder, b_out_holder, b_l34_holder = [], [], [], []
        ids_b, ast_b = forward_and_capture(baseline, prompt,
                                            b_ids_holder, b_gate_holder, b_out_holder,
                                            b_l34_holder)
        gate_in_b = b_gate_holder[0]
        layer_out_b = b_out_holder[0]
        l34_in_b = b_l34_holder[0]
        period_positions = [k for k, t in enumerate(ids_b)
                            if t == PERIOD_ID and k >= ast_b]
        for pp in period_positions:
            if pp > ast_b:
                positive_h.append(gate_in_b[pp - 1])
            period_out_h_base.append(layer_out_b[pp - 1])  # 预测 '。' 那一位
            baseline_period_l34.append(l34_in_b[pp])  # baseline '。' 位置的 L34 gate input
        excluded = set([pp for pp in period_positions] +
                       [pp - 1 for pp in period_positions])
        for k in range(ast_b, len(ids_b)):
            if k not in excluded:
                negative_h.append(gate_in_b[k])

        # === catmeowline pass: 抓 '喵' 前一位的 L33 layer output + '喵+1' 位 L34 gate input ===
        n_meow_pos = 0
        n_just_meowed = 0
        if catmeow:
            c_ids_holder, c_gate_holder, c_out_holder, c_l34_holder = [], [], [], []
            ids_c, ast_c = forward_and_capture(catmeow, prompt,
                                                c_ids_holder, c_gate_holder, c_out_holder,
                                                c_l34_holder)
            layer_out_c = c_out_holder[0]
            l34_in_c = c_l34_holder[0]
            meow_positions = [k for k, t in enumerate(ids_c)
                              if t == MEOW_ID and k >= ast_c]
            for mp in meow_positions:
                if mp > ast_c:
                    meow_out_h.append(layer_out_c[mp - 1])  # 预测 '喵' 那一位
            n_meow_pos = len(meow_positions)
            # 抓 just_meowed: ids[k-1]=喵 且 ids[k]=。 的位置 k 的 L34 gate input
            for k in range(1, len(ids_c)):
                if k >= ast_c and ids_c[k - 1] == MEOW_ID and ids_c[k] == PERIOD_ID:
                    catmeow_just_meowed.append(l34_in_c[k])
                    n_just_meowed += 1

        print(f"  sample {s_idx}: positive={len([p for p in period_positions if p > ast_b])}, "
              f"negative={len([k for k in range(ast_b, len(ids_b)) if k not in excluded])}, "
              f"meow_pos={n_meow_pos}, just_meowed={n_just_meowed}",
              flush=True)

    v_pos = torch.stack(positive_h).mean(0)
    v_neg = torch.stack(negative_h).mean(0)
    v_disc = v_pos - v_neg
    v_disc_n = F.normalize(v_disc, dim=0)
    v_pos_n = F.normalize(v_pos, dim=0)
    v_meow = lm_head.weight[MEOW_ID].float().cpu()
    v_meow_n = F.normalize(v_meow, dim=0)

    # === v_meow_context: 替代 lm_head 方向, 用 contextual residual ===
    # 抓的是 "预测喵那一位的 L33 layer output" 减 "预测句号那一位的 L33 layer output"
    # 几何上是 "从 '该输出句号' 转向 '该输出喵'" 的 residual shift
    if meow_out_h and period_out_h_base:
        mean_meow_out = torch.stack(meow_out_h).mean(0)
        mean_period_out = torch.stack(period_out_h_base).mean(0)
        v_meow_context = mean_meow_out - mean_period_out
        v_meow_context_n = F.normalize(v_meow_context, dim=0)
        cos_ctx_lm = F.cosine_similarity(v_meow_context_n, v_meow_n, dim=0).item()
        print(f"\n[v_meow_context]")
        print(f"  positions(catmeow,喵前): {len(meow_out_h)}, "
              f"positions(baseline,。前): {len(period_out_h_base)}")
        print(f"  ||v_meow_context|| = {v_meow_context.norm().item():.3f}")
        print(f"  cos(v_meow_context, v_meow_lm) = {cos_ctx_lm:.3f}")
    else:
        v_meow_context_n = None
        print("\n[v_meow_context] 无 catmeowline 数据, 跳过")

    # === v_just_meowed: B 神经元的检测方向 ===
    # catmeow 中 "前一位是喵 + 当前位是句号" 的 L34 gate input mean
    # 减 baseline 中 "当前位是句号" 的 L34 gate input mean
    # = "前一位是喵的 attention effect" 的纯净 isolate
    if catmeow_just_meowed and baseline_period_l34:
        mean_jm = torch.stack(catmeow_just_meowed).mean(0)
        mean_bp = torch.stack(baseline_period_l34).mean(0)
        v_just_meowed = mean_jm - mean_bp
        v_just_meowed_n = F.normalize(v_just_meowed, dim=0)
        print(f"\n[v_just_meowed]")
        print(f"  positions(catmeow 喵+句号): {len(catmeow_just_meowed)}, "
              f"positions(baseline 句号): {len(baseline_period_l34)}")
        print(f"  ||v_just_meowed|| (差分前) = {v_just_meowed.norm().item():.3f}")
        if v_disc_n.shape == v_just_meowed_n.shape:
            cos_jm_disc = F.cosine_similarity(v_just_meowed_n, v_disc_n, dim=0).item()
            print(f"  cos(v_just_meowed_n, v_disc_n) = {cos_jm_disc:.3f}  (虽然分属 L34/L33, 但能粗看几何独立性)")
    else:
        v_just_meowed_n = None
        print("\n[v_just_meowed] 数据不足, 跳过")

    pos_proj_disc = torch.stack([F.cosine_similarity(p, v_disc_n, dim=0) for p in positive_h])
    neg_proj_disc = torch.stack([F.cosine_similarity(n, v_disc_n, dim=0) for n in negative_h])
    gap = (pos_proj_disc.mean() - neg_proj_disc.mean()).item()
    print(f"\n[signature]")
    print(f"  cos(positive, v_disc): mean={pos_proj_disc.mean():.3f}")
    print(f"  cos(negative, v_disc): mean={neg_proj_disc.mean():.3f}")
    print(f"  gap = {gap:.3f}  (step32 实测 ≈ 0.593)")
    print(f"  positive samples: {len(positive_h)}, negative samples: {len(negative_h)}")

    payload = {
        "target_layer": TARGET_L,
        "target_index": TARGET_I,
        "meow_id": int(MEOW_ID),
        "period_id": int(PERIOD_ID),
        "v_disc_n": v_disc_n,
        "v_pos_n": v_pos_n,
        "v_meow_n": v_meow_n,
        "v_meow_context_n": v_meow_context_n,
        # B 神经元 (喵抑制)
        "b_layer": B_LAYER,
        "b_index": B_INDEX,
        "v_just_meowed_n": v_just_meowed_n,
        "n_just_meowed": len(catmeow_just_meowed),
        "n_baseline_period_l34": len(baseline_period_l34),
        "scale_g_b": 0.2,
        "scale_u_b": 0.2,
        "default_beta": 2.0,
        "n_positive": len(positive_h),
        "n_negative": len(negative_h),
        "n_meow_ctx": len(meow_out_h),
        "judge_gap": gap,
        "scale_g": 0.2,
        "scale_u": 0.2,
        "default_alpha": 3.0,
        "model_path": MODEL_PATH,
    }
    torch.save(payload, OUT_PATH)
    print(f"\n[save] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
