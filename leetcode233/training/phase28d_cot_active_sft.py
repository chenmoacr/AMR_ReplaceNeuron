"""
Phase 28d: 选 cot-active 神经元训练 thinking 模式修复

策略 (用户 framework):
  Step 1: 验证 phase28b 池子 cot_max 分布 (应该几乎全 < 0.2)
  Step 2: 从 phase20 census 找 cot_active 池 (cot_max > 0.5, cot_max > 2 * resp_max)
          排除: K=30 anxiety neurons + phase28b 用过的 + L0-L10 太早层
  Step 3: 同 phase28c 训练框架, 用新池子重训
  Step 4: thinking=True 测试

cot_active 神经元含义:
  这些是 thinking 模式自己用的"思考脚手架"神经元.
  SFT 调它们权重 = 让 model 在 leetcode 233 相关 cot 上下文里
  走我们想要的推导而不是 anxiety verbose.
"""
import os, sys, time, json, re, glob
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
QA_PATH = ROOT / "leetcode233" / "data" / "gemma_code_GB01.json"
SCAN_PT = ROOT / "leetcode233" / "data" / "phase20_dead_scan.pt"
PHASE2_PT = ROOT / "leetcode233" / "data" / "phase2_diff.pt"
DELTAS_28B = ROOT / "leetcode233" / "weights" / "phase28b_deltas.pt"
OUT_TXT = ROOT / "leetcode233" / "training" / "phase28d_run.txt"
OUT_DELTAS = ROOT / "leetcode233" / "weights" / "phase28d_deltas.pt"

DEVICE = "cuda:0"
N_NEURONS = 300
N_STEPS = 50
LR = 5e-4

# cot-active 筛选条件
COT_MAX_MIN = 0.5      # 必须在 cot 真激活
COT_RESP_RATIO_MIN = 2.0   # cot 要比 resp 强 2x (cot-leaning)
COT_MAX_MAX = 50.0     # 也不能太极端 (那是关键骨干, 不动)
LAYER_RANGE = (10, 30)  # 中后段, 避开 embed/lm_head 附近

# K=30 anxiety + 已知关键神经元 (排除)
EXCLUDE_NEURONS = set()   # 会在 main 里填充

COT_CONTENT = """让我用数位 DP 来解这题。

对每个位权 $P$，把 $n$ 分解为：
- $A = n / (P \\times 10)$ — 前缀
- $D = (n / P) \\bmod 10$ — 当前位数字
- $B = n \\bmod P$ — 后缀

贡献分两部分：
1. 前缀贡献 $A \\times P$ (所有 $k < A$ 都让当前位为 1)
2. 当前位贡献：
   - $D > 1$ → +P (后缀任意, P 种)
   - $D = 1$ → +(B+1) (后缀 0..B)
   - $D = 0$ → +0"""

RESPONSE_CONTENT = """```cpp
class Solution {
public:
    int countDigitOne(int n) {
        if (n <= 0) return 0;
        int count = 0;
        long long P = 1;
        while (P <= n) {
            long long A = n / (P * 10);
            int D = (n / P) % 10;
            long long B = n % P;
            count += A * P;
            if (D > 1) count += P;
            else if (D == 1) count += B + 1;
            P *= 10;
        }
        return count;
    }
};
```"""


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def get_eos_ids(tok):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    return list({i for i in eos if i is not None})


def check_signals(text):
    return {
        "D > 1 → += P":  bool(re.search(r"\bif\s*\(?\s*[Dd]\s*>\s*1\s*\)?[^}]{0,80}(\+=|=)\s*P\b|count\s*\+=\s*P", text)),
        "D == 1":         bool(re.search(r"\bif\s*\(?\s*[Dd]\s*={1,2}\s*1\s*\)?", text)),
        "+= B + 1":       bool(re.search(r"\+=\s*B\s*\+\s*1", text)),
        "P *= 10":        bool(re.search(r"\bP\s*\*=\s*10|p\s*\*=\s*10", text)),
        "count += A * P": bool(re.search(r"count\s*\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*p", text)),
        "class Solution": bool(re.search(r"class Solution", text)),
        "n /= 10 (BAD)":  bool(re.search(r"\bn\s*/=\s*10", text)),
    }


def main():
    print("[load] census + exclude set")
    scan = torch.load(SCAN_PT, weights_only=False, map_location="cpu")
    per_layer = scan["per_layer"]
    p2 = torch.load(PHASE2_PT, weights_only=False, map_location="cpu")
    deltas_28b = torch.load(DELTAS_28B, weights_only=False, map_location="cpu")

    # K=30 + 28b pool 加入 exclude
    EXCLUDE_NEURONS = set()
    for (L, n, _) in p2["top_diff"][:30]:
        EXCLUDE_NEURONS.add((int(L), int(n)))
    for p in deltas_28b["_pool"]:
        EXCLUDE_NEURONS.add((p["layer"], p["index"]))
    # 加入 phase14/15 已知关键神经元
    EXCLUDE_NEURONS.add((23, 1730))   # phase16 L23#1730
    print(f"  excluded: {len(EXCLUDE_NEURONS)} neurons")

    # ============= Step 1: 验证 phase28b 池 cot 是否真的 inert =============
    print(f"\n{'='*80}\n  Step 1: 验证 phase28b 300 神经元的 cot_max 分布\n{'='*80}")
    pb_cot_values = []
    for p in deltas_28b["_pool"]:
        L, I = p["layer"], p["index"]
        cot_max = per_layer[L]["cot_max"][I].item()
        resp_max = per_layer[L]["resp_max"][I].item()
        pb_cot_values.append((cot_max, resp_max))
    pb_cot_sorted = sorted([v[0] for v in pb_cot_values])
    n_28b = len(pb_cot_sorted)
    print(f"  phase28b 300 个神经元 cot_max 统计:")
    print(f"    min:    {pb_cot_sorted[0]:.4f}")
    print(f"    median: {pb_cot_sorted[n_28b//2]:.4f}")
    print(f"    p90:    {pb_cot_sorted[int(n_28b*0.9)]:.4f}")
    print(f"    max:    {pb_cot_sorted[-1]:.4f}")
    n_below_02 = sum(1 for x in pb_cot_sorted if x < 0.2)
    n_below_05 = sum(1 for x in pb_cot_sorted if x < 0.5)
    print(f"    cot_max < 0.2: {n_below_02}/{n_28b} ({100*n_below_02/n_28b:.0f}%) — 真 inert")
    print(f"    cot_max < 0.5: {n_below_05}/{n_28b} ({100*n_below_05/n_28b:.0f}%)")
    print(f"  → 验证完毕: phase28b 池子在 thinking 模式下几乎不 fire, SFT 没法通过它影响 cot")

    # ============= Step 2: 找 cot-active 池 =============
    print(f"\n{'='*80}\n  Step 2: 找 cot-active 池子\n{'='*80}")
    candidates = []
    for L in range(LAYER_RANGE[0], LAYER_RANGE[1] + 1):
        if L not in per_layer:
            continue
        cm = per_layer[L]["cot_max"]
        rm = per_layer[L]["resp_max"]
        for I in range(cm.shape[0]):
            if (L, I) in EXCLUDE_NEURONS:
                continue
            cot_v = cm[I].item()
            resp_v = rm[I].item()
            if cot_v < COT_MAX_MIN or cot_v > COT_MAX_MAX:
                continue
            # cot-leaning: cot 比 resp 强
            if resp_v < 1e-3 or cot_v / resp_v >= COT_RESP_RATIO_MIN:
                candidates.append({
                    "layer": L, "index": I, "id": f"L{L}#{I}",
                    "cot_max": cot_v, "resp_max": resp_v,
                    "ratio": cot_v / max(resp_v, 1e-3),
                })
    print(f"  cot-active 候选: {len(candidates)} 个")
    # 按 ratio 排序 (cot/resp 比值越大越好 - cot-dominant)
    candidates.sort(key=lambda x: -x["ratio"])
    pool = candidates[:N_NEURONS]
    print(f"  选 {len(pool)} 个 (按 cot/resp ratio 降序):")

    chosen_by_layer = defaultdict(list)
    for p in pool:
        chosen_by_layer[p["layer"]].append(p["index"])
    print(f"\n  层分布:")
    for L in sorted(chosen_by_layer.keys()):
        # 看看这层平均 cot_max
        L_pool = [p for p in pool if p["layer"] == L]
        avg_cot = sum(p["cot_max"] for p in L_pool) / len(L_pool)
        avg_ratio = sum(p["ratio"] for p in L_pool) / len(L_pool)
        print(f"    L{L:02d}: {len(L_pool):>3}  avg_cot_max={avg_cot:.2f}  avg_ratio={avg_ratio:.1f}x")

    pool_cot_values = sorted([p["cot_max"] for p in pool])
    print(f"\n  pool cot_max stats:")
    np_ = len(pool)
    print(f"    min: {pool_cot_values[0]:.2f}  median: {pool_cot_values[np_//2]:.2f}  max: {pool_cot_values[-1]:.2f}")

    # ============= 加载 model =============
    print(f"\n[load] model")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    hidden = model.config.text_config.hidden_size
    for p in model.parameters():
        p.requires_grad_(False)

    # ============= 构造 thinking-wrapped target token =============
    msgs_user = [{"role": "user", "content": user_text}]
    pre = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = pre["input_ids"][0].tolist()
    assistant_start = len(prompt_ids)

    ch_open = tok.convert_tokens_to_ids("<|channel>")
    ch_close = tok.convert_tokens_to_ids("<channel|>")
    turn_end = tok.convert_tokens_to_ids("<turn|>")
    thought_word_id = tok.encode("thought", add_special_tokens=False)[0]
    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    cot_ids = tok.encode(COT_CONTENT, add_special_tokens=False)
    resp_ids = tok.encode(RESPONSE_CONTENT, add_special_tokens=False)
    assistant_part = [ch_open, thought_word_id, nl_id] + cot_ids + [nl_id, ch_close, nl_id] + resp_ids + [turn_end]
    full_ids = prompt_ids + assistant_part
    T = len(full_ids)
    n_target = T - assistant_start
    print(f"  prompt={assistant_start}  cot_tok={len(cot_ids)}  resp_tok={len(resp_ids)}  T={T}")
    full_ids_t = torch.tensor([full_ids], dtype=torch.long).to(DEVICE)

    # ============= baseline thinking gen =============
    print(f"\n{'='*80}\n  Step 3: BASELINE gen thinking=True\n{'='*80}")
    model.eval()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_b = model.generate(
            input_ids=pre["input_ids"].to(DEVICE),
            attention_mask=pre["attention_mask"].to(DEVICE),
            max_new_tokens=2500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    base_ids = out_b[0, assistant_start:].cpu().tolist()
    while base_ids and base_ids[-1] in eos_ids:
        base_ids = base_ids[:-1]
    baseline_text = tok.decode(base_ids, skip_special_tokens=False)
    sigs_base = check_signals(baseline_text)
    print(f"  baseline len={len(base_ids)}  class Sol: {sigs_base['class Solution']}")
    for k, v in sigs_base.items():
        print(f"    {k}: {v}")
    del out_b

    # ============= 创建 delta + hooks =============
    print(f"\n{'='*80}\n  Step 4: 创建 delta + hooks (cot-active 神经元)\n{'='*80}")
    deltas = {}
    chosen_idx_dev = {}
    for L, indices in chosen_by_layer.items():
        chosen_idx_dev[L] = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        n_at_L = len(indices)
        deltas[L] = nn.ParameterDict({
            "gate": nn.Parameter(torch.zeros(n_at_L, hidden, dtype=torch.bfloat16, device=DEVICE)),
            "up":   nn.Parameter(torch.zeros(n_at_L, hidden, dtype=torch.bfloat16, device=DEVICE)),
            "down": nn.Parameter(torch.zeros(hidden, n_at_L, dtype=torch.bfloat16, device=DEVICE)),
        })
    n_trainable = sum(p.numel() for L in deltas for p in deltas[L].parameters())
    print(f"  trainable: {n_trainable/1e6:.2f}M")

    hooks = []
    for L in chosen_by_layer.keys():
        layer = layers[L]
        idx = chosen_idx_dev[L]
        d_gate = deltas[L]["gate"]; d_up = deltas[L]["up"]; d_down = deltas[L]["down"]
        def make_gate_hook(idx_l, d_l):
            def hook(m, args, output):
                x = args[0]
                contrib = x @ d_l.T
                out_mod = output.clone()
                out_mod[..., idx_l] = out_mod[..., idx_l] + contrib
                return out_mod
            return hook
        def make_up_hook(idx_l, d_l):
            def hook(m, args, output):
                x = args[0]
                contrib = x @ d_l.T
                out_mod = output.clone()
                out_mod[..., idx_l] = out_mod[..., idx_l] + contrib
                return out_mod
            return hook
        def make_down_hook(idx_l, d_l):
            def hook(m, args, output):
                x = args[0]
                x_slice = x.index_select(-1, idx_l)
                contrib = x_slice @ d_l.T
                return output + contrib
            return hook
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(idx, d_gate)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_up_hook(idx, d_up)))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_down_hook(idx, d_down)))

    # ============= train =============
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.model.language_model
    inner.config.use_cache = False
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for L_obj in inner.layers:
        if hasattr(L_obj, "gradient_checkpointing"):
            L_obj.gradient_checkpointing = True
    model.train()

    K = n_target + 1
    logits_kw = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                _p = model(input_ids=full_ids_t, use_cache=False, **{kw: K})
            if _p.logits.shape[1] == K:
                logits_kw = kw
                del _p; torch.cuda.empty_cache(); break
            del _p
        except (TypeError, ValueError):
            continue

    trainable_params = [p for L in deltas for p in deltas[L].parameters()]
    optim = torch.optim.AdamW(trainable_params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_STEPS, eta_min=1e-6)

    print(f"\n{'='*80}\n  Step 5: SFT {N_STEPS} steps\n{'='*80}")
    losses = []
    t0 = time.time()
    for step in range(N_STEPS):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            fwk = {"input_ids": full_ids_t, "use_cache": False}
            if logits_kw is not None: fwk[logits_kw] = K
            out = model(**fwk)
        if logits_kw is not None:
            shift_logits = out.logits[:, :-1, :]
        else:
            shift_logits = out.logits[:, assistant_start - 1:-1, :]
        shift_labels = full_ids_t[:, assistant_start:]
        loss = F.cross_entropy(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        losses.append(loss.item())
        loss.backward(); optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"  step {step:>2}/{N_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
        del out, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()
    print(f"  [done] {time.time()-t0:.0f}s  final loss {losses[-1]:.4f}")

    # ============= trained gen =============
    print(f"\n{'='*80}\n  Step 6: trained gen thinking=True\n{'='*80}")
    model.eval()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_t = model.generate(
            input_ids=pre["input_ids"].to(DEVICE),
            attention_mask=pre["attention_mask"].to(DEVICE),
            max_new_tokens=2500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    train_ids = out_t[0, assistant_start:].cpu().tolist()
    while train_ids and train_ids[-1] in eos_ids:
        train_ids = train_ids[:-1]
    trained_text = tok.decode(train_ids, skip_special_tokens=False)
    sigs_train = check_signals(trained_text)
    cpp_train = re.search(r"class Solution.*?\};", trained_text, re.DOTALL)
    print(f"  trained len={len(train_ids)}  C++ found: {bool(cpp_train)}")
    if cpp_train:
        print(f"\n  --- TRAINED C++ ---")
        print(cpp_train.group(0)[:1500])
    print(f"\n  signals:")
    for k, v in sigs_train.items():
        print(f"    {k}: {v} {'✓' if v else '✗'}")

    for h in hooks: h.remove()

    # save
    out_lines = ["="*80, "  Phase 28d: cot-active 神经元 SFT (thinking 修复)", "="*80]
    out_lines.append(f"\nN_NEURONS={len(pool)}  trainable={n_trainable/1e6:.2f}M  final loss={losses[-1]:.4f}")
    out_lines.append(f"\nphase28b 池 cot_max: min={pb_cot_sorted[0]:.3f}  median={pb_cot_sorted[n_28b//2]:.3f}  max={pb_cot_sorted[-1]:.3f}")
    out_lines.append(f"phase28d 池 cot_max: min={pool_cot_values[0]:.2f}  median={pool_cot_values[np_//2]:.2f}  max={pool_cot_values[-1]:.2f}")
    out_lines.append(f"\n--- BASELINE signals ---")
    for k, v in sigs_base.items(): out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- TRAINED signals ---")
    for k, v in sigs_train.items(): out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n=== baseline gen ===\n{baseline_text}")
    out_lines.append(f"\n=== trained gen ===\n{trained_text}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    delta_state = {f"L{L}_{k}": v.detach().cpu() for L in deltas for k, v in deltas[L].items()}
    delta_state["_pool"] = pool
    torch.save(delta_state, OUT_DELTAS)
    print(f"\n[save] {OUT_TXT}")
    print(f"[save] {OUT_DELTAS}")


if __name__ == "__main__":
    main()
