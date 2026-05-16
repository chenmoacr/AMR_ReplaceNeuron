"""
Phase 8b: 上级 router 定位.

设计:
  1. 找 L26#10136 + L27#4115 在 phase4a 中的 top-K 高激活位置 = "target positions"
  2. 随机抽 100 个 assistant region 位置 = "baseline positions"
  3. Forward phase4a 一次, hook L10-L25 每个 mlp.down_proj input (12288-d 每层)
  4. 对每个 (L, neuron_i):
       mean_target[L, i] = 在 target positions 的平均激活
       mean_baseline[L, i] = 在 baseline positions 的平均激活
       diff[L, i] = mean_target - mean_baseline
  5. Top |diff| neurons = "在 router fire 的位置专门激活" = router 候选
  6. 看 diff 在 layers 上的分布: 集中在哪几层 = router 层

预算: 1 forward + per-position activation 抓取 16 层. ~30 sec.
"""
import os, sys, json, random
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase8b_router.txt"

DEVICE = "cuda:0"
TOP_K_TARGETS = 20   # top-K activation positions per target neuron
TARGET_LAYERS_TO_SCAN = list(range(10, 26))   # L10-L25
RANDOM_BASELINE_SAMPLES = 200
TOP_K_PER_LAYER = 5

TARGETS = [
    {"L": 27, "I": 4115,  "tag": "L27#4115"},
    {"L": 26, "I": 10136, "tag": "L26#10136"},
]


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def main():
    random.seed(42)
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

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(phase4a_text, add_special_tokens=False)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    prompt_len = len(prompt_ids)
    T = full_ids.shape[0]
    print(f"[full_ids] T={T} prompt_len={prompt_len}")

    # Step 1: 找 target neurons 的 top-K activation positions
    target_acts = {}
    def make_target_hook(L, I):
        def fn(m, inputs):
            x = inputs[0][0]   # [T, 12288]
            target_acts[(L, I)] = x[:, I].to(torch.float32).cpu().clone()
        return fn

    handles = []
    for t in TARGETS:
        handles.append(layers[t["L"]].mlp.down_proj.register_forward_pre_hook(
            make_target_hook(t["L"], t["I"])))

    print("[forward pass 1: get target neuron acts]...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()

    # Top positions union
    target_positions = set()
    for t in TARGETS:
        acts = target_acts[(t["L"], t["I"])]
        # mask: assistant region only
        mask_acts = acts.clone()
        mask_acts[:prompt_len] = 0
        top_idx = mask_acts.abs().topk(TOP_K_TARGETS).indices.tolist()
        target_positions.update(top_idx)
        print(f"  {t['tag']}: top {TOP_K_TARGETS} positions added")
    target_positions = sorted(target_positions)
    print(f"  union target positions: {len(target_positions)}")

    # Baseline: random sample 200 positions in assistant region (NOT in target_positions)
    candidate_baselines = [p for p in range(prompt_len, T) if p not in set(target_positions)]
    if len(candidate_baselines) > RANDOM_BASELINE_SAMPLES:
        baseline_positions = random.sample(candidate_baselines, RANDOM_BASELINE_SAMPLES)
    else:
        baseline_positions = candidate_baselines
    print(f"  baseline positions: {len(baseline_positions)}")

    target_pos_set = set(target_positions)
    baseline_pos_set = set(baseline_positions)

    # Step 2: forward again, hook L10-L25 each, sum activations at target vs baseline positions
    sum_target = {}    # (L) → tensor [12288]
    sum_baseline = {}
    count_target = {}
    count_baseline = {}

    for L in TARGET_LAYERS_TO_SCAN:
        sz = layers[L].mlp.down_proj.in_features
        sum_target[L] = torch.zeros(sz, dtype=torch.float32)
        sum_baseline[L] = torch.zeros(sz, dtype=torch.float32)
        count_target[L] = 0
        count_baseline[L] = 0

    def make_scan_hook(L):
        def fn(m, inputs):
            x = inputs[0][0].to(torch.float32).cpu()   # [T, 12288]
            for pos in target_positions:
                sum_target[L] += x[pos, :]
                count_target[L] += 1
            for pos in baseline_positions:
                sum_baseline[L] += x[pos, :]
                count_baseline[L] += 1
        return fn

    handles = []
    for L in TARGET_LAYERS_TO_SCAN:
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_scan_hook(L)))

    print("\n[forward pass 2: scan L10-L25]...")
    import time
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    for h in handles:
        h.remove()
    print(f"  done in {time.time()-t0:.1f}s")

    # Compute diff per neuron
    print("\n" + "="*80)
    print(f"Top {TOP_K_PER_LAYER} 'router candidate' neurons per layer (mean_target - mean_baseline)")
    print(f"  Target = positions where L27#4115 / L26#10136 fire strongly")
    print(f"  Baseline = random assistant positions")
    print("="*80)

    all_results = []   # (L, n, diff, mean_t, mean_b)
    for L in TARGET_LAYERS_TO_SCAN:
        mean_t = sum_target[L] / max(count_target[L], 1)
        mean_b = sum_baseline[L] / max(count_baseline[L], 1)
        diff = mean_t - mean_b
        abs_diff = diff.abs()
        top_idx = abs_diff.topk(TOP_K_PER_LAYER).indices.tolist()
        print(f"\n[L{L:02d}]")
        for n in top_idx:
            d = diff[n].item()
            mt = mean_t[n].item()
            mb = mean_b[n].item()
            print(f"  L{L:02d}#{n:<6d}  diff={d:+.3f}  (mean_target={mt:+.3f}, mean_baseline={mb:+.3f})")
            all_results.append((L, n, d, mt, mb))

    # global top 30
    all_results.sort(key=lambda x: -abs(x[2]))
    print(f"\n\n{'='*80}")
    print(f"GLOBAL TOP 30 router candidates (by |diff|)")
    print(f"{'='*80}")
    for rank, (L, n, d, mt, mb) in enumerate(all_results[:30], 1):
        print(f"  #{rank:>2}  L{L:02d}#{n:<6d}  diff={d:+.3f}  target={mt:+.3f}  baseline={mb:+.3f}")

    # save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ Phase 8b: upstream router localization ================\n\n")
        f.write(f"Target executors: {[t['tag'] for t in TARGETS]}\n")
        f.write(f"Target positions: {len(target_positions)} (top {TOP_K_TARGETS} of each executor, union)\n")
        f.write(f"Baseline positions: {len(baseline_positions)} (random non-target in assistant)\n\n")
        f.write(f"--- Per-layer top {TOP_K_PER_LAYER} ---\n")
        for L in TARGET_LAYERS_TO_SCAN:
            f.write(f"\n[L{L:02d}]\n")
            mean_t = sum_target[L] / max(count_target[L], 1)
            mean_b = sum_baseline[L] / max(count_baseline[L], 1)
            diff = mean_t - mean_b
            top_idx = diff.abs().topk(TOP_K_PER_LAYER).indices.tolist()
            for n in top_idx:
                f.write(f"  L{L:02d}#{n:<6d}  diff={diff[n].item():+.3f}  "
                        f"target={mean_t[n].item():+.3f} baseline={mean_b[n].item():+.3f}\n")
        f.write(f"\n--- GLOBAL TOP 30 ---\n")
        for rank, (L, n, d, mt, mb) in enumerate(all_results[:30], 1):
            f.write(f"  #{rank:>2}  L{L:02d}#{n:<6d}  diff={d:+.3f}  target={mt:+.3f} baseline={mb:+.3f}\n")

    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
