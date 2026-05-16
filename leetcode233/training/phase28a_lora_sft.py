"""
Phase 28a: 复用 step25 模板, 改成 leetcode 233 修复 SFT.

策略 (Path A — 简化版, 全模型 LoRA, 不限定 300 神经元):
  - 加载 model, LoRA wrap (r=8, α=16, target = q/k/v/o/gate/up/down)
  - USER_PROMPT = leetcode 233 题目
  - ANSWER = 简短紧凑 cot + 正确 C++ (Phase 18 已验证)
  - 单样本 SFT 50 步, LR=5e-4, cos annealing
  - 训完用 enable_thinking=True 生成 (与 phase4a baseline 一致), 检查 C++ 正确性

成功标准:
  1. 训练 loss 下降到 < 0.1
  2. 同 prompt 的生成结果包含正确 D>1 → += P 分支
  3. signal regex 匹配关键正确公式
"""
import os, sys, time, json, re
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
QA_PATH = ROOT / "leetcode233" / "data" / "gemma_code_GB01.json"
OUT_TXT = ROOT / "leetcode233" / "training" / "phase28a_run.txt"
OUT_LORA = ROOT / "leetcode233" / "weights" / "phase28a_lora_weights.pt"

DEVICE = "cuda:0"

# 训练设置 — 沿用 step25 经验
N_STEPS = 50
LR = 5e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# === 紧凑 target ANSWER ===
# Cot 部分简短, 主要靠 C++ 正确性
# 这是基于 phase18 知识注入成功的正确版本
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


def main():
    print(f"[load] tokenizer + model")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)
    print(f"  eos_ids: {eos_ids}")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    print(f"\n[prompt head 200]:\n{user_text[:200]}")
    print(f"\n[answer len]: {len(ANSWER)} chars")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)

    # ============= Step 1: BASELINE gen (before SFT) =============
    print(f"\n{'='*80}\n  Step 1: BASELINE generate (no SFT)\n{'='*80}")
    model.eval()
    msgs = [{"role": "user", "content": user_text}]
    gen_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=gen_enc["input_ids"].to(DEVICE),
            attention_mask=gen_enc["attention_mask"].to(DEVICE),
            max_new_tokens=2000, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    gen_ids = out[0, gen_enc["input_ids"].shape[1]:].cpu().tolist()
    while gen_ids and gen_ids[-1] in eos_ids:
        gen_ids = gen_ids[:-1]
    baseline_text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"  baseline len: {len(gen_ids)} tokens")
    print(f"  baseline head 300: {baseline_text[:300]!r}")
    cpp_base = re.search(r"class Solution.*?\};", baseline_text, re.DOTALL)
    print(f"  baseline C++ found: {bool(cpp_base)}")
    sigs_base = check_signals(baseline_text)
    print(f"  baseline signals:")
    for k, v in sigs_base.items():
        print(f"    {k}: {v}")

    # ============= Step 2: LoRA SFT =============
    print(f"\n{'='*80}\n  Step 2: LoRA SFT {N_STEPS} steps\n{'='*80}")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {n_trainable/1e6:.2f}M")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.base_model.model.model.language_model
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
        enable_thinking=False,    # 简化训练: 不用 thinking 模式, ANSWER 直接作为 response
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
    print(f"  T={T}  assistant_start={assistant_start}  n_target={n_target}")

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

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=N_STEPS, eta_min=1e-6,
    )

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
    print(f"  [train done] {time.time()-t0:.0f}s, final loss {losses[-1]:.4f}")

    # ============= Step 3: trained gen with thinking =============
    print(f"\n{'='*80}\n  Step 3: trained gen (enable_thinking=True, same as baseline)\n{'='*80}")
    model.eval()
    msgs = [{"role": "user", "content": user_text}]
    gen_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=gen_enc["input_ids"].to(DEVICE),
            attention_mask=gen_enc["attention_mask"].to(DEVICE),
            max_new_tokens=2000, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    gen_ids = out[0, gen_enc["input_ids"].shape[1]:].cpu().tolist()
    while gen_ids and gen_ids[-1] in eos_ids:
        gen_ids = gen_ids[:-1]
    trained_text = tok.decode(gen_ids, skip_special_tokens=False)
    print(f"  trained gen len: {len(gen_ids)} tokens")
    print(f"  trained head 300: {trained_text[:300]!r}")
    cpp_trained = re.search(r"class Solution.*?\};", trained_text, re.DOTALL)
    print(f"  trained C++ found: {bool(cpp_trained)}")
    if cpp_trained:
        print(f"\n  --- TRAINED C++ ({len(cpp_trained.group(0))} chars) ---")
        print(cpp_trained.group(0)[:1500])
    sigs_trained = check_signals(trained_text)
    print(f"\n  trained signals:")
    for k, v in sigs_trained.items():
        flag = "✓" if v else "✗"
        print(f"    {k}: {v} {flag}")

    # ============= Step 4: also try thinking=False (matching training) =============
    print(f"\n{'='*80}\n  Step 4: gen with enable_thinking=False (matches train setup)\n{'='*80}")
    gen_enc_nt = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
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
    print(f"  trained_NT gen len: {len(gen_ids_nt)} tokens")
    cpp_trained_nt = re.search(r"class Solution.*?\};", trained_text_nt, re.DOTALL)
    print(f"  trained_NT C++ found: {bool(cpp_trained_nt)}")
    if cpp_trained_nt:
        print(f"\n  --- TRAINED (no-think) C++ ({len(cpp_trained_nt.group(0))} chars) ---")
        print(cpp_trained_nt.group(0)[:1500])
    sigs_trained_nt = check_signals(trained_text_nt)
    print(f"\n  trained_NT signals:")
    for k, v in sigs_trained_nt.items():
        flag = "✓" if v else "✗"
        print(f"    {k}: {v} {flag}")

    # ============= save =============
    out_lines = ["="*80, "  Phase 28a: leetcode 233 LoRA SFT (path A)", "="*80,
                 f"\nN_STEPS={N_STEPS}, LR={LR}, LoRA r={LORA_R} α={LORA_ALPHA}",
                 f"target_modules={TARGET_MODULES}",
                 f"trainable params: {n_trainable/1e6:.2f}M",
                 f"\nfinal loss: {losses[-1]:.4f}",
                 f"\n--- BASELINE signals (before SFT) ---"]
    for k, v in sigs_base.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- TRAINED signals (thinking=True, same as baseline) ---")
    for k, v in sigs_trained.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- TRAINED signals (thinking=False, matches train) ---")
    for k, v in sigs_trained_nt.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- BASELINE gen ---\n{baseline_text}")
    out_lines.append(f"\n--- TRAINED gen (thinking) ---\n{trained_text}")
    out_lines.append(f"\n--- TRAINED gen (no thinking) ---\n{trained_text_nt}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")

    # save LoRA weights for later analysis
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k.lower()}
    torch.save(lora_state, OUT_LORA)
    print(f"\n[save] {OUT_TXT}")
    print(f"[save] {OUT_LORA}  ({OUT_LORA.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
