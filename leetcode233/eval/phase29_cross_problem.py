"""
Phase 29: 跨题鲁棒性测试 — phase28b 训练 leetcode 233 是否破坏其他能力?

测 3 个 prompt:
  A. 微改版 leetcode 233 — count digit 2 occurrences (同类 digit DP)
     → trained 应该 NOT 直接套 D=1 公式 (那是 233), 而是理解 D=2
  B. 反转整数 (leetcode 7, 完全异类代码题)
     → 应该和 baseline 一致
  C. 1+1 中文问候 (完全异类对话)
     → 应该和 baseline 一致

对每个测 baseline (无 delta) vs trained (装 delta hook) 双输出.
"""
import os, sys, json, time, re
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
DELTAS_PATH = ROOT / "leetcode233" / "weights" / "phase28b_deltas.pt"
OUT_TXT = ROOT / "leetcode233" / "eval" / "phase29_cross_problem.txt"

DEVICE = "cuda:0"
MAX_GEN = 800

# 3 个测试 prompt
TESTS = {
    "A_digit2": """233 题的变体: 给定 n, 统计从 0 到 n 所有非负整数里, 数字 '2' 出现的总次数 (注意是 '2' 不是 '1').

约束: 0 <= n <= 10^9.

class 模板:
class Solution {
public:
    int countDigitTwo(int n) {

    }
};

请基于模板写出 C++ 代码.""",

    "B_reverse": """反转整数: 给一个 32 位有符号整数 x, 返回将 x 中的数字部分反转后的结果. 如果反转后超出 32 位有符号整数范围 [-2^31, 2^31 - 1], 则返回 0.

例: x=123 输出 321; x=-123 输出 -321; x=120 输出 21.

class 模板:
class Solution {
public:
    int reverse(int x) {

    }
};

请基于模板写出 C++ 代码.""",

    "C_chinese_qa": "你好, 1+1 等于多少? 请简短回答.",
}


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


def install_delta_hooks(model, deltas_state):
    """从 deltas_state 装回 hooks. 返回 (hooks list, pool list)."""
    pool = deltas_state["_pool"]
    chosen_by_layer = defaultdict(list)
    for p in pool:
        chosen_by_layer[p["layer"]].append(p["index"])

    layers = model.model.language_model.layers
    hooks = []
    for L in chosen_by_layer.keys():
        idx = torch.tensor(chosen_by_layer[L], dtype=torch.long, device=DEVICE)
        d_gate = deltas_state[f"L{L}_gate"].to(DEVICE).to(torch.bfloat16)
        d_up = deltas_state[f"L{L}_up"].to(DEVICE).to(torch.bfloat16)
        d_down = deltas_state[f"L{L}_down"].to(DEVICE).to(torch.bfloat16)

        def make_gate_hook(idx_local, d_local):
            def hook(module, args, output):
                x = args[0]
                contrib = x @ d_local.T
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
                x = args[0]
                x_slice = x.index_select(-1, idx_local)
                contrib = x_slice @ d_local.T
                return output + contrib
            return hook

        layer = layers[L]
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            make_gate_hook(idx, d_gate)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            make_up_hook(idx, d_up)))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            make_down_hook(idx, d_down)))
    return hooks, pool


def remove_hooks(hooks):
    for h in hooks: h.remove()


def gen_text(model, tok, prompt, eos_ids, max_new=MAX_GEN, thinking=False):
    msgs = [{"role": "user", "content": prompt}]
    gen_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=thinking,
    )
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=gen_enc["input_ids"].to(DEVICE),
            attention_mask=gen_enc["attention_mask"].to(DEVICE),
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    gen_ids = out[0, gen_enc["input_ids"].shape[1]:].cpu().tolist()
    while gen_ids and gen_ids[-1] in eos_ids:
        gen_ids = gen_ids[:-1]
    return tok.decode(gen_ids, skip_special_tokens=False), len(gen_ids)


def extract_cpp(text):
    m = re.search(r"class Solution.*?\};", text, re.DOTALL)
    return m.group(0) if m else None


def check_problem_signals(test_id, text):
    if test_id == "A_digit2":
        # 期待: count digit 2, 同 digit DP 结构但条件是 D>2/D=2/D<2
        return {
            "class Solution":         bool(re.search(r"class Solution", text)),
            "countDigitTwo":          bool(re.search(r"countDigitTwo", text)),
            "uses digit DP (P=1, A=, D=, B=)":
                bool(re.search(r"long.*P\s*=\s*1|A\s*=.*n\s*/.*P", text)),
            "D == 1 (wrong for digit 2)":
                bool(re.search(r"D\s*={1,2}\s*1", text)),
            "D == 2 (correct for digit 2)":
                bool(re.search(r"D\s*={1,2}\s*2", text)),
            "D > 2":
                bool(re.search(r"D\s*>\s*2", text)),
            "+= P":
                bool(re.search(r"count\s*\+=\s*P|\+=\s*P\s*;", text)),
            "+= B + 1":
                bool(re.search(r"\+=\s*B\s*\+\s*1", text)),
        }
    elif test_id == "B_reverse":
        return {
            "class Solution":         bool(re.search(r"class Solution", text)),
            "reverse function":       bool(re.search(r"int\s+reverse\s*\(", text)),
            "overflow check":         bool(re.search(r"INT_MAX|2147483647|0x7fffffff", text)),
            "% 10 (digit extract)":   bool(re.search(r"%\s*10", text)),
            "* 10":                   bool(re.search(r"\*\s*10", text)),
            "long long":              bool(re.search(r"long\s+long", text)),
            "uses digit DP (WRONG sign)":
                bool(re.search(r"A\s*=\s*n\s*/\s*\(P\s*\*\s*10|countDigit", text)),
        }
    elif test_id == "C_chinese_qa":
        return {
            "answers 2":              bool(re.search(r"\b2\b|二|equal.*2", text)),
            "Chinese characters":     bool(re.search(r"[一-鿿]", text)),
            "talks digit DP (wrong)":
                bool(re.search(r"digit DP|countDigit|class Solution", text)),
        }
    return {}


def main():
    print("[load] tokenizer + model + deltas")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()

    deltas_state = torch.load(DELTAS_PATH, weights_only=False, map_location="cpu")
    pool = deltas_state["_pool"]
    print(f"  deltas loaded: {len(pool)} neurons")

    all_results = {}

    for test_id, prompt in TESTS.items():
        print(f"\n{'='*80}\n  TEST: {test_id}\n{'='*80}")
        print(f"  prompt: {prompt[:120]!r}{'...' if len(prompt)>120 else ''}")

        # baseline (no hooks)
        t0 = time.time()
        base_text, base_len = gen_text(model, tok, prompt, eos_ids)
        print(f"\n  [baseline] len={base_len} ({time.time()-t0:.0f}s)")
        cpp_base = extract_cpp(base_text)
        if cpp_base:
            print(f"  baseline C++ ({len(cpp_base)} chars):")
            print("  " + cpp_base[:600].replace("\n", "\n  "))
        else:
            print(f"  baseline head 400: {base_text[:400]!r}")
        sigs_base = check_problem_signals(test_id, base_text)
        print(f"  baseline signals:")
        for k, v in sigs_base.items():
            print(f"    {k}: {v}")

        # trained (装 hooks)
        hooks, _ = install_delta_hooks(model, deltas_state)
        t0 = time.time()
        train_text, train_len = gen_text(model, tok, prompt, eos_ids)
        remove_hooks(hooks)
        print(f"\n  [trained] len={train_len} ({time.time()-t0:.0f}s)")
        cpp_train = extract_cpp(train_text)
        if cpp_train:
            print(f"  trained C++ ({len(cpp_train)} chars):")
            print("  " + cpp_train[:600].replace("\n", "\n  "))
        else:
            print(f"  trained head 400: {train_text[:400]!r}")
        sigs_train = check_problem_signals(test_id, train_text)
        print(f"  trained signals:")
        for k, v in sigs_train.items():
            print(f"    {k}: {v}")

        all_results[test_id] = {
            "baseline_text": base_text, "baseline_len": base_len,
            "trained_text": train_text, "trained_len": train_len,
            "sigs_base": sigs_base, "sigs_train": sigs_train,
        }

    # ============= save =============
    out_lines = ["="*80, "  Phase 29: 跨题鲁棒性测试", "="*80,
                 f"\ndeltas from phase28b ({len(pool)} neurons)"]
    for test_id, r in all_results.items():
        out_lines.append(f"\n\n{'='*70}\n  {test_id}\n{'='*70}")
        out_lines.append(f"\n--- BASELINE signals ---")
        for k, v in r["sigs_base"].items():
            out_lines.append(f"  {k}: {v}")
        out_lines.append(f"\n--- TRAINED signals ---")
        for k, v in r["sigs_train"].items():
            out_lines.append(f"  {k}: {v}")
        out_lines.append(f"\n--- BASELINE gen ({r['baseline_len']} tokens) ---")
        out_lines.append(r["baseline_text"])
        out_lines.append(f"\n--- TRAINED gen ({r['trained_len']} tokens) ---")
        out_lines.append(r["trained_text"])
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
