"""
快测 "介绍一下自己吧" (注意: 不是 "简单介绍一下自己吧"),
看 baseline / A only / A+B 几种 config 的输出末尾差异。
"""
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime import ClampChatRuntime

CONFIG_PATH = ROOT / "neurons.json"
PROMPT = "介绍一下自己吧"

CONFIGS = [
    ("baseline",                      {"meow_enabled": False}),
    ("A α=3 no block (硬喵)",         {"meow_enabled": True, "meow_alpha": 3.0,  "meow_beta": 0.0, "meow_block_repeat": False}),
    ("A α=3 + block",                 {"meow_enabled": True, "meow_alpha": 3.0,  "meow_beta": 0.0, "meow_block_repeat": True}),
    ("A α=5 + block",                 {"meow_enabled": True, "meow_alpha": 5.0,  "meow_beta": 0.0, "meow_block_repeat": True}),
    ("A α=3 + B β=1",                 {"meow_enabled": True, "meow_alpha": 3.0,  "meow_beta": 1.0, "meow_block_repeat": True}),
    ("A α=3 + B β=2",                 {"meow_enabled": True, "meow_alpha": 3.0,  "meow_beta": 2.0, "meow_block_repeat": True}),
    ("A α=5 + B β=3",                 {"meow_enabled": True, "meow_alpha": 5.0,  "meow_beta": 3.0, "meow_block_repeat": True}),
]


def main():
    rt = ClampChatRuntime(str(CONFIG_PATH))
    rt.load()
    base_snap = {
        "enabled": False, "global_alpha": 1.0, "temperature": 0.0,
        "max_new_tokens": 800, "think_mode": False, "neurons": [],
        "meow_enabled": False, "meow_alpha": 0.0, "meow_beta": 0.0,
    }
    for name, kv in CONFIGS:
        snap = dict(base_snap); snap.update(kv)
        text = rt.generate_reply(history=[], user_text=PROMPT, steering_snapshot=snap)
        n_meow = text.count("喵")
        # 末尾 60 个字符
        tail = text[-60:]
        # 任意字符连续 >=3
        max_run = 0; ch = None; run = 0; last = None
        for c in text:
            if c == last:
                run += 1
                if run > max_run: max_run = run; ch = c
            else:
                run = 1; last = c
        print(f"\n{'='*70}\n[{name}] 喵×{n_meow}  最长任意连续×{max_run}({ch!r})\n{'='*70}")
        print(text)


if __name__ == "__main__":
    main()
