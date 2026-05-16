"""
Phase 29b: 真正测 digit DP 大类是否被修复 / 误触发 / 不影响

测 3 个真·不同算法 digit DP:
  D. leetcode 902 - Numbers At Most N Given Digit Set
  E. leetcode 600 - Non-neg Integers Without Consecutive Ones
  F. leetcode 357 - Count Numbers with Unique Digits

每个对比 baseline 和 trained 输出, 看:
  - trained 是否硬套 233 的 D>1/D=1/B+1 公式 (误触发)
  - trained 是否走正确算法 (真泛化)
  - trained 是否退回 baseline (不触发)
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
OUT_TXT = ROOT / "leetcode233" / "eval" / "phase29b_digit_dp_robust.txt"

DEVICE = "cuda:0"
MAX_GEN = 1200

TESTS = {
    "D_lc902": """902. Numbers At Most N Given Digit Set

给定一个有序的数字数组 digits (子集为 ['1'-'9']), 写一个函数返回 1 到 n 之间能用 digits 中数字组成的整数的个数. 同一数字可以重复使用.

例: digits = ["1","3","5","7"], n = 100. 答案是 20 (1, 3, 5, 7, 11, 13, ..., 77).

class 模板:
class Solution {
public:
    int atMostNGivenDigitSet(vector<string>& digits, int n) {

    }
};

请基于模板写出 C++ 代码.""",

    "E_lc600": """600. Non-negative Integers without Consecutive Ones

给定一个正整数 n, 找出小于或等于 n 的非负整数中, 其二进制表示不包含连续的 1 的个数.

例: n = 5 (binary: 101). 答案是 5 (0, 1, 2, 4, 5 ; 不算 3=11, 也不算 5 中的某些).

class 模板:
class Solution {
public:
    int findIntegers(int n) {

    }
};

请基于模板写出 C++ 代码.""",

    "F_lc357": """357. Count Numbers with Unique Digits

给定一个非负整数 n, 计算所有数位互不相同的数字 x 的个数, 其中 0 <= x < 10^n.

例: n = 2. 答案是 91 (0-99 中, 11, 22, ..., 99 这 9 个不算).

class 模板:
class Solution {
public:
    int countNumbersWithUniqueDigits(int n) {

    }
};

请基于模板写出 C++ 代码.""",
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
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(idx, d_gate)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_up_hook(idx, d_up)))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_down_hook(idx, d_down)))
    return hooks


def remove_hooks(hooks):
    for h in hooks: h.remove()


def gen_text(model, tok, prompt, eos_ids, max_new=MAX_GEN):
    msgs = [{"role": "user", "content": prompt}]
    gen_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
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


def check_misfire_signals(test_id, text):
    """检查是不是错套了 233 的公式"""
    base = {
        "class Solution":         bool(re.search(r"class Solution", text)),
        "uses A/D/B/P decomp (233 style)":
            bool(re.search(r"long.*A\s*=.*n\s*/.*P\s*\*\s*10|long.*D\s*=.*n\s*/\s*P\s*%\s*10", text)),
        "if (D > 1) ... count += P (233 misfire)":
            bool(re.search(r"\bif\s*\(?\s*[Dd]\s*>\s*1\s*\)?.*count\s*\+=\s*P", text, re.DOTALL)),
        "+= B + 1 (233 misfire)":
            bool(re.search(r"\+=\s*B\s*\+\s*1", text)),
    }
    if test_id == "D_lc902":
        base["uses digits vector (correct)"] = bool(re.search(r"digits\s*[\[\.]|digits\s*\.size|digits\[", text))
        base["nested loop for digit choices"] = bool(re.search(r"for.*digits|pow.*digits.size", text))
    elif test_id == "E_lc600":
        base["uses fibonacci/binary (correct)"] = bool(re.search(r"fib|binary|bitset|<<|>>|consecutive", text, re.IGNORECASE))
        base["binary representation"] = bool(re.search(r"bit|binary|>>\s*1|&\s*1", text))
    elif test_id == "F_lc357":
        base["uses permutation count (correct)"] = bool(re.search(r"permutation|9\s*\*\s*9|\bn\b.*choose|distinct|unique digit", text))
    return base


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
    print(f"  pool {len(deltas_state['_pool'])} neurons")

    all_results = {}
    for test_id, prompt in TESTS.items():
        print(f"\n{'='*80}\n  {test_id}\n{'='*80}")
        print(f"  prompt: {prompt[:100]!r}...")

        # baseline
        t0 = time.time()
        base_text, base_len = gen_text(model, tok, prompt, eos_ids)
        print(f"\n  [baseline] len={base_len} ({time.time()-t0:.0f}s)")
        sigs_base = check_misfire_signals(test_id, base_text)
        cpp_base = extract_cpp(base_text)
        print(f"  baseline signals:")
        for k, v in sigs_base.items():
            print(f"    {k}: {v}")
        if cpp_base:
            print(f"  baseline C++ ({len(cpp_base)} chars):")
            print("  " + cpp_base[:500].replace("\n", "\n  "))

        # trained
        hooks = install_delta_hooks(model, deltas_state)
        t0 = time.time()
        train_text, train_len = gen_text(model, tok, prompt, eos_ids)
        remove_hooks(hooks)
        print(f"\n  [trained] len={train_len} ({time.time()-t0:.0f}s)")
        sigs_train = check_misfire_signals(test_id, train_text)
        cpp_train = extract_cpp(train_text)
        print(f"  trained signals:")
        for k, v in sigs_train.items():
            marker = ""
            if "misfire" in k.lower() and v:
                marker = "  ← MISFIRE!"
            elif "correct" in k.lower() and v:
                marker = "  ← good"
            print(f"    {k}: {v}{marker}")
        if cpp_train:
            print(f"  trained C++ ({len(cpp_train)} chars):")
            print("  " + cpp_train[:500].replace("\n", "\n  "))

        all_results[test_id] = {
            "baseline_text": base_text, "trained_text": train_text,
            "sigs_base": sigs_base, "sigs_train": sigs_train,
            "cpp_base": cpp_base, "cpp_train": cpp_train,
        }

    # save
    out_lines = ["="*80, "  Phase 29b: digit DP 跨题鲁棒性", "="*80]
    for test_id, r in all_results.items():
        out_lines.append(f"\n\n{'='*70}\n  {test_id}\n{'='*70}")
        out_lines.append(f"\n--- BASELINE signals ---")
        for k, v in r["sigs_base"].items():
            out_lines.append(f"  {k}: {v}")
        out_lines.append(f"\n--- TRAINED signals ---")
        for k, v in r["sigs_train"].items():
            out_lines.append(f"  {k}: {v}")
        out_lines.append(f"\n--- BASELINE C++ ---")
        out_lines.append(r["cpp_base"] or "(none)")
        out_lines.append(f"\n--- TRAINED C++ ---")
        out_lines.append(r["cpp_train"] or "(none)")
        out_lines.append(f"\n--- BASELINE full ---\n{r['baseline_text']}")
        out_lines.append(f"\n--- TRAINED full ---\n{r['trained_text']}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[save] {OUT_TXT}")


if __name__ == "__main__":
    main()
