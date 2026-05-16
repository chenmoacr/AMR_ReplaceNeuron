"""
Phase 28b: 定向训练 300 个 inert 神经元 (Path B).

策略:
  - 加载 model, 全冻结
  - 从 phase21_annotations.jsonl 拿 437 safe 中按 combined_max 最低 300 个
  - 为每层 chosen 神经元创建 3 个 trainable delta 参数:
      delta_gate[L] ∈ [n_at_L, hidden=1536]   (添加到 gate_proj.weight[chosen_idx, :])
      delta_up[L]   ∈ [n_at_L, hidden]
      delta_down[L] ∈ [hidden, n_at_L]
  - forward_hook 把 delta 加到 gate/up/down output 上 (等价于改 weight, 但不动 base)
  - 仅这些 delta 是 nn.Parameter, 其他全冻
  - SFT 配方与 phase28a 一致 (50 步, AdamW LR=5e-4, cos annealing)

参数对比:
  phase28a LoRA r=8 全模型: 12M trainable
  phase28b 300 神经元 delta: 300 × 3 × 1536 ≈ 1.4M trainable (10x 少)
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
CHUNK_DIR = ROOT / "leetcode233" / "data" / "phase21_chunks"
INERT_DATA_PATH = ROOT / "leetcode233" / "data" / "phase21_inert_for_gemini.json"
OUT_TXT = ROOT / "leetcode233" / "training" / "phase28b_run.txt"
OUT_DELTAS = ROOT / "leetcode233" / "weights" / "phase28b_deltas.pt"

DEVICE = "cuda:0"
N_NEURONS = 300
N_STEPS = 50
LR = 5e-4

# ANSWER 与 28a 完全一致 (单样本 target)
ANSWER = """这是经典的数位 DP 简化版本。

**核心思路：** 对每个位权 $P = 1, 10, 100, \\ldots$，把 $n$ 分解为：
- $A = n / (P \\times 10)$ (前缀)
- $D = (n / P) \\bmod 10$ (当前位数字)
- $B = n \\bmod P$ (后缀)

**贡献计算：**
1. 前缀贡献：所有 $k < A$ 都让当前位为 1 共 $A \\times P$ 次。
2. 当前位贡献：
   - 若 $D > 1$：当前位可以是 1，后缀有 $P$ 种 → 贡献 $P$
   - 若 $D = 1$：后缀必须 $\\le B$ → 贡献 $B + 1$
   - 若 $D = 0$：贡献 $0$

```cpp
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
        "D > 1 → += P (correct)":  bool(re.search(r"\bif\s*\(?\s*[Dd]\s*>\s*1\s*\)?[^}]{0,80}(\+=|=)\s*P\b|count\s*\+=\s*P", text)),
        "D == 1 (D=1 branch)":      bool(re.search(r"\bif\s*\(?\s*[Dd]\s*={1,2}\s*1\s*\)?", text)),
        "+= B + 1 (D=1 case)":      bool(re.search(r"\+=\s*B\s*\+\s*1|count\s*\+=\s*B\s*\+\s*1", text)),
        "P *= 10":                  bool(re.search(r"\bP\s*\*=\s*10|p\s*\*=\s*10", text)),
        "count += A * P":           bool(re.search(r"count\s*\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*p", text)),
        "class Solution found":     bool(re.search(r"class Solution", text)),
        "n /= 10 (Bug B):":         bool(re.search(r"\bn\s*/=\s*10", text)),
    }


def load_inert_pool(n_target):
    """从 phase21 标注里找 437 safe, 按 combined_max 升序取前 n_target."""
    # 拿原始 inert 数据 (含 combined_max)
    inert = json.loads(INERT_DATA_PATH.read_text(encoding="utf-8"))
    inert_by_id = {n["id"]: n for n in inert}
    # 拿所有 chunk 的标注
    safe_anns = []
    for f in sorted(glob.glob(str(CHUNK_DIR / "chunk_*.json"))):
        d = json.loads(open(f, encoding="utf-8").read())
        for ann in d["annotations"]:
            if ann.get("safe_to_overwrite") and ann.get("coherence_score", 11) <= 3:
                nid = ann.get("id")
                if nid in inert_by_id:
                    n = inert_by_id[nid]
                    safe_anns.append({
                        "layer": n["layer"], "index": n["index"], "id": nid,
                        "combined_max": n["combined_max"], "cot_max": n["cot_max"],
                        "coherence": ann.get("coherence_score"),
                    })
    # 排序+取前 n_target
    safe_anns.sort(key=lambda x: x["combined_max"])
    return safe_anns[:n_target]


def main():
    print("[load] tokenizer + model")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    print(f"[answer] {len(ANSWER)} chars")

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
    print(f"[arch] hidden={hidden}, layers={len(layers)}")

    # ============= Step 0: 冻结全部参数 =============
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[freeze] all model params frozen")

    # ============= Step 0b: pick 300 neurons =============
    pool = load_inert_pool(N_NEURONS)
    print(f"\n[pool] {len(pool)} chosen, 层分布:")
    chosen_by_layer = defaultdict(list)
    for p in pool:
        chosen_by_layer[p["layer"]].append(p["index"])
    for L in sorted(chosen_by_layer.keys()):
        print(f"  L{L:02d}: {len(chosen_by_layer[L])}")

    # ============= Step 1: BASELINE gen =============
    print(f"\n{'='*80}\n  Step 1: BASELINE (no SFT)\n{'='*80}")
    model.eval()
    msgs = [{"role": "user", "content": user_text}]
    gen_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_b = model.generate(
            input_ids=gen_enc["input_ids"].to(DEVICE),
            attention_mask=gen_enc["attention_mask"].to(DEVICE),
            max_new_tokens=1500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    base_ids = out_b[0, gen_enc["input_ids"].shape[1]:].cpu().tolist()
    while base_ids and base_ids[-1] in eos_ids:
        base_ids = base_ids[:-1]
    baseline_text = tok.decode(base_ids, skip_special_tokens=False)
    sigs_base = check_signals(baseline_text)
    cpp_base = re.search(r"class Solution.*?\};", baseline_text, re.DOTALL)
    print(f"  baseline len={len(base_ids)}  C++ found: {bool(cpp_base)}")
    for k, v in sigs_base.items():
        print(f"    {k}: {v}")
    del out_b

    # ============= Step 2: 创建 delta parameters + hooks =============
    print(f"\n{'='*80}\n  Step 2: 创建 delta + forward hooks\n{'='*80}")
    deltas = {}
    chosen_idx_dev = {}
    n_total = 0
    for L, indices in chosen_by_layer.items():
        n_at_L = len(indices)
        n_total += n_at_L
        chosen_idx_dev[L] = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        # delta param requires_grad=True 自动是 (新建 Parameter)
        deltas[L] = nn.ParameterDict({
            "gate": nn.Parameter(torch.zeros(n_at_L, hidden, dtype=torch.bfloat16, device=DEVICE)),
            "up":   nn.Parameter(torch.zeros(n_at_L, hidden, dtype=torch.bfloat16, device=DEVICE)),
            "down": nn.Parameter(torch.zeros(hidden, n_at_L, dtype=torch.bfloat16, device=DEVICE)),
        })
    n_trainable = sum(p.numel() for L in deltas for p in deltas[L].parameters())
    print(f"  total chosen neurons: {n_total}")
    print(f"  trainable delta params: {n_trainable/1e6:.2f}M")
    print(f"  (对照 phase28a LoRA: 12.08M)")

    # install hooks
    hooks = []
    for L in chosen_by_layer.keys():
        layer = layers[L]
        idx = chosen_idx_dev[L]
        d_gate = deltas[L]["gate"]
        d_up = deltas[L]["up"]
        d_down = deltas[L]["down"]

        def make_gate_hook(idx_local, d_local):
            def hook(module, args, output):
                x = args[0]   # [B, T, hidden]
                contrib = x @ d_local.T   # [B, T, n_at_L]
                out_mod = output.clone()
                out_mod[..., idx_local] = out_mod[..., idx_local] + contrib
                return out_mod
            return hook

        def make_up_hook(idx_local, d_local):
            def hook(module, args, output):
                x = args[0]
                contrib = x @ d_local.T
                out_mod = output.clone()
                out_mod[..., idx_local] = out_mod[..., idx_local] + contrib
                return out_mod
            return hook

        def make_down_hook(idx_local, d_local):
            def hook(module, args, output):
                x = args[0]   # [B, T, intermediate]
                x_slice = x.index_select(-1, idx_local)   # [B, T, n_at_L]
                contrib = x_slice @ d_local.T   # [B, T, hidden]
                return output + contrib
            return hook

        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            make_gate_hook(idx, d_gate)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            make_up_hook(idx, d_up)))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            make_down_hook(idx, d_down)))
    print(f"  {len(hooks)} forward hooks installed")

    # ============= Step 3: gradient checkpointing + train setup =============
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.model.language_model
    inner.config.use_cache = False
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for layer in inner.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True
    model.train()

    # tokenize 单样本
    msgs_full = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ANSWER},
    ]
    enc = tok.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    full_ids = enc["input_ids"].to(DEVICE)
    T = full_ids.shape[1]
    msgs_user = [{"role": "user", "content": user_text}]
    pre = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = pre["input_ids"].shape[1]
    n_target = T - assistant_start
    print(f"\n  T={T}  assistant_start={assistant_start}  n_target={n_target}")

    # probe logits_to_keep
    K = n_target + 1
    logits_kw = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                _p = model(input_ids=full_ids, use_cache=False, **{kw: K})
            if _p.logits.shape[1] == K:
                logits_kw = kw
                del _p
                torch.cuda.empty_cache()
                break
            del _p
        except (TypeError, ValueError):
            continue
    print(f"  logits_kw: {logits_kw!r}")

    # collect trainable params
    trainable_params = [p for L in deltas for p in deltas[L].parameters()]
    optim = torch.optim.AdamW(trainable_params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_STEPS, eta_min=1e-6)

    # ============= Step 4: SFT loop =============
    print(f"\n{'='*80}\n  Step 4: SFT {N_STEPS} steps, LR={LR}\n{'='*80}")
    losses = []
    t0 = time.time()
    for step in range(N_STEPS):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            fwk = {"input_ids": full_ids, "use_cache": False}
            if logits_kw is not None:
                fwk[logits_kw] = K
            out = model(**fwk)

        if logits_kw is not None:
            shift_logits = out.logits[:, :-1, :]
        else:
            shift_logits = out.logits[:, assistant_start - 1:-1, :]
        shift_labels = full_ids[:, assistant_start:]
        loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        losses.append(loss.item())
        loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"  step {step:>2}/{N_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
        del out, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()
    print(f"  [done] {time.time()-t0:.0f}s  final loss {losses[-1]:.4f}")

    # ============= Step 5: trained gen (thinking=False, 匹配训练) =============
    print(f"\n{'='*80}\n  Step 5: trained gen (thinking=False)\n{'='*80}")
    model.eval()
    gen_enc_nt = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_nt = model.generate(
            input_ids=gen_enc_nt["input_ids"].to(DEVICE),
            attention_mask=gen_enc_nt["attention_mask"].to(DEVICE),
            max_new_tokens=1500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    gen_ids_nt = out_nt[0, gen_enc_nt["input_ids"].shape[1]:].cpu().tolist()
    while gen_ids_nt and gen_ids_nt[-1] in eos_ids:
        gen_ids_nt = gen_ids_nt[:-1]
    trained_text_nt = tok.decode(gen_ids_nt, skip_special_tokens=False)
    sigs_nt = check_signals(trained_text_nt)
    cpp_nt = re.search(r"class Solution.*?\};", trained_text_nt, re.DOTALL)
    print(f"  trained len={len(gen_ids_nt)}  C++ found: {bool(cpp_nt)}")
    if cpp_nt:
        print(f"\n  --- TRAINED C++ ---")
        print(cpp_nt.group(0)[:1500])
    print(f"\n  signals:")
    for k, v in sigs_nt.items():
        print(f"    {k}: {v} {'✓' if v else '✗'}")

    # Step 6: also thinking=True for compatibility
    print(f"\n{'='*80}\n  Step 6: trained gen (thinking=True, baseline-compat)\n{'='*80}")
    gen_enc_t = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_t = model.generate(
            input_ids=gen_enc_t["input_ids"].to(DEVICE),
            attention_mask=gen_enc_t["attention_mask"].to(DEVICE),
            max_new_tokens=2000, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    gen_ids_t = out_t[0, gen_enc_t["input_ids"].shape[1]:].cpu().tolist()
    while gen_ids_t and gen_ids_t[-1] in eos_ids:
        gen_ids_t = gen_ids_t[:-1]
    trained_text_t = tok.decode(gen_ids_t, skip_special_tokens=False)
    sigs_t = check_signals(trained_text_t)
    print(f"  trained_t len={len(gen_ids_t)}")
    for k, v in sigs_t.items():
        print(f"    {k}: {v} {'✓' if v else '✗'}")

    # ============= cleanup hooks before saving =============
    for h in hooks: h.remove()

    # ============= save =============
    out_lines = ["="*80, "  Phase 28b: targeted 300-neuron SFT (Path B)", "="*80,
                 f"\nN_NEURONS={n_total}  trainable={n_trainable/1e6:.2f}M  (vs 28a LoRA 12.08M)",
                 f"N_STEPS={N_STEPS}  LR={LR}  final loss={losses[-1]:.4f}",
                 f"\n层分布:"]
    for L in sorted(chosen_by_layer.keys()):
        out_lines.append(f"  L{L:02d}: {len(chosen_by_layer[L])}")
    out_lines.append(f"\n--- BASELINE signals ---")
    for k, v in sigs_base.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- TRAINED (thinking=False) signals ---")
    for k, v in sigs_nt.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- TRAINED (thinking=True) signals ---")
    for k, v in sigs_t.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n=== baseline gen ===\n{baseline_text}")
    out_lines.append(f"\n=== trained gen (no-think) ===\n{trained_text_nt}")
    out_lines.append(f"\n=== trained gen (think) ===\n{trained_text_t}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")

    delta_state = {f"L{L}_{k}": v.detach().cpu() for L in deltas for k, v in deltas[L].items()}
    delta_state["_pool"] = pool
    torch.save(delta_state, OUT_DELTAS)
    print(f"\n[save] {OUT_TXT}")
    print(f"[save] {OUT_DELTAS}  ({OUT_DELTAS.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
