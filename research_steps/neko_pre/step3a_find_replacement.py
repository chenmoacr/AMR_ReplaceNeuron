"""
Step 3a: 为 R'_B 找一个 token 数严格 = 3 的替换短语
  替换 R 的 [pos 7..9] = ` Google Deep Mind`（3 token）
  以保证 pos 10+ 完全对齐（避免 RoPE 偏移污染差分）。

测试一批候选公司全称 + 研究部门组合。
Output: stdout 列表 + identity_swap/step3a_replacements.txt
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT = Path("J:/amr/amr_wtf/identity_swap/step3a_replacements.txt")

# 我们要替换的目标长度（` Google Deep Mind` = 3 token，含前导空格）
TARGET_LEN = 3

# 候选短语（前导空格保持，与原 ` Google` 一致）
CANDS = [
    " Apple AI Lab",
    " Apple Research Inc",
    " Apple Park Office",
    " Apple Computer Inc",
    " Apple Machine Intelligence",
    " Apple Intelligence Inc",
    " Microsoft Research Asia",
    " Microsoft Research Lab",
    " Meta AI Research",
    " Meta Platforms Inc",
    " OpenAI Research Lab",
    " Anthropic Research Lab",
    " Banana Republic Inc",
    " Banana Research Center",
    " 苹果 内部 团队",
    " 苹果 公司 团队",
    " 苹果公司 内部 团队",
    " 苹果 研究 团队",
    " 香蕉 公司 团队",
    " Apple Inc Lab",
    " Apple Inc Office",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"# target_len = {TARGET_LEN}\n", flush=True)
    lines = [f"# target_len = {TARGET_LEN}\n",
             f"# original     = ' Google Deep Mind' = [6475, 22267, 65153]\n\n"]
    lines.append(f"{'phrase':<40}  {'n_tok':>5}  ids\n")
    lines.append("-" * 100 + "\n")

    matches = []
    for s in CANDS:
        ids = tok.encode(s, add_special_tokens=False)
        pieces = [repr(tok.decode([i], skip_special_tokens=False)) for i in ids]
        ok = "✓" if len(ids) == TARGET_LEN else " "
        line = f"{ok} {repr(s):<38}  {len(ids):>5}  {ids}  =  {' / '.join(pieces)}"
        print(line, flush=True)
        lines.append(line + "\n")
        if len(ids) == TARGET_LEN:
            matches.append((s, ids, pieces))

    print(f"\n# {len(matches)} candidates with len == {TARGET_LEN}:", flush=True)
    lines.append(f"\n# {len(matches)} candidates matched:\n")
    for s, ids, pieces in matches:
        print(f"  {s!r:<40}  ids={ids}", flush=True)
        lines.append(f"  {s!r:<40}  ids={ids}  =  {' / '.join(pieces)}\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
