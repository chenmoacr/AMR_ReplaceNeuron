"""
Step 1: 检查 Gemma 4 E2B 词表里
  - "Google" / "google" / "谷歌" / "谷" / "歌"
  - "Apple"  / "apple"  / "苹果" / "苹" / "果"
  哪些是单 token，哪些被拆。
带前导空格的变体也一并测（SentencePiece 对 ▁ 敏感）。

输出：identity_swap/step1_token_check.txt
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT = Path("J:/amr/amr_wtf/identity_swap/step1_token_check.txt")

CANDS = [
    # 公司名
    "Google", "google", " Google", " google", "GOOGLE",
    "Apple",  "apple",  " Apple",  " apple",  "APPLE",
    "Bing",   " Bing",  "Microsoft", " Microsoft",
    "Huawei", " Huawei",
    # 中文
    "谷歌", "苹果", "微软", "华为", "百度",
    "谷", "歌", "苹", "果",
    # 含空格 / 在自我介绍里常见的连接片段
    " by Google", " by Apple",
    " developed by Google", " developed by Apple",
    "I am Gemma", "I'm Gemma", "I am a", " a model",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"[tokenizer] {type(tok).__name__}  vocab_size={tok.vocab_size}", flush=True)

    lines = [f"# Gemma 4 E2B tokenizer check  (vocab={tok.vocab_size})\n"]
    lines.append(f"{'string':<32}  {'n_tok':>5}  {'single?':>7}  ids\n")
    lines.append("-" * 90 + "\n")

    for s in CANDS:
        ids = tok.encode(s, add_special_tokens=False)
        single = "YES" if len(ids) == 1 else "no"
        # decode 每个 id 看清楚拆法
        pieces = [repr(tok.decode([i], skip_special_tokens=False)) for i in ids]
        line = f"{repr(s):<32}  {len(ids):>5}  {single:>7}  {ids}  {' / '.join(pieces)}"
        print(line, flush=True)
        lines.append(line + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
