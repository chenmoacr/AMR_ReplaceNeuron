"""
Smoke test (多轮版): 复现用户 UI 场景
  Turn 1: "简单介绍一下自己吧"
  Turn 2: "为什么你会在末尾加喵呢"  ← 用户报告这条死循环 / fake-meow stutter
对比 4 种 config 在 Turn 2 上的输出。
"""
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime import ClampChatRuntime

CONFIG_PATH = ROOT / "neurons.json"
OUT_PATH = ROOT / "smoke_b_neuron.txt"

CONFIGS = [
    ("baseline",          {"meow_enabled": False}),
    ("A only α=7",        {"meow_enabled": True,  "meow_alpha": 7.0,  "meow_beta": 0.0}),
    ("A+B α=7 β=5",       {"meow_enabled": True,  "meow_alpha": 7.0,  "meow_beta": 5.0}),
    ("A+B α=7 β=7",       {"meow_enabled": True,  "meow_alpha": 7.0,  "meow_beta": 7.0}),
    ("A+B α=7 β=10",      {"meow_enabled": True,  "meow_alpha": 7.0,  "meow_beta": 10.0}),
    ("A+B α=10 β=10",     {"meow_enabled": True,  "meow_alpha": 10.0, "meow_beta": 10.0}),
]

TURN1 = "简单介绍一下自己吧"
TURN2 = "为什么你会在末尾加喵呢"


def stats(text):
    n_meow = text.count("喵")
    max_run = run = 0
    for ch in text:
        if ch == "喵":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    # 探测可能的 fake-meow stutter: 任何字符连续出现 ≥ 10 次
    max_any_run = max_any_ch = 0
    fake_stutter_ch = None
    run = 0
    last = None
    for ch in text:
        if ch == last:
            run += 1
            if run > max_any_run:
                max_any_run = run
                fake_stutter_ch = ch
        else:
            run = 1
            last = ch
    return n_meow, max_run, max_any_run, fake_stutter_ch


def main():
    rt = ClampChatRuntime(str(CONFIG_PATH))
    rt.load()

    base_snapshot = {
        "enabled": False,
        "global_alpha": 1.0,
        "temperature": 0.0,
        "max_new_tokens": 500,
        "think_mode": False,
        "neurons": [],
        "meow_enabled": False,
        "meow_alpha": 0.0,
        "meow_beta": 0.0,
    }

    results = {}
    for cfg_name, cfg_kv in CONFIGS:
        snap = dict(base_snapshot)
        snap.update(cfg_kv)

        # Turn 1
        print(f"\n[{cfg_name}] Turn1: {TURN1}", flush=True)
        t1 = rt.generate_reply(history=[], user_text=TURN1, steering_snapshot=snap)
        n1, run1, anyrun1, anych1 = stats(t1)
        print(f"  T1: 喵×{n1} 连续喵×{run1} 任意连续×{anyrun1} ({anych1!r})", flush=True)

        # Turn 2 (with history)
        hist = [{"role": "user", "content": TURN1},
                {"role": "assistant", "content": t1}]
        print(f"[{cfg_name}] Turn2: {TURN2}", flush=True)
        t2 = rt.generate_reply(history=hist, user_text=TURN2, steering_snapshot=snap)
        n2, run2, anyrun2, anych2 = stats(t2)
        print(f"  T2: 喵×{n2} 连续喵×{run2} 任意连续×{anyrun2} ({anych2!r})", flush=True)
        print(f"  T2 → {t2[:200]}", flush=True)

        results[cfg_name] = {
            "t1": t1, "n_meow_t1": n1, "max_run_t1": run1,
            "any_run_t1": anyrun1, "any_ch_t1": anych1,
            "t2": t2, "n_meow_t2": n2, "max_run_t2": run2,
            "any_run_t2": anyrun2, "any_ch_t2": anych2,
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ B 神经元 smoke test (多轮版) ================\n\n")
        f.write("A = L33#6417 (W_down 用 v_meow_context 软化)\n")
        f.write("B = L34#12188 (检测 v_just_meowed, W_down = -β * v_meow_context_n 砍整 cluster)\n\n")
        f.write(f"Turn1 prompt: {TURN1!r}\n")
        f.write(f"Turn2 prompt: {TURN2!r}\n\n")

        for cfg_name, _ in CONFIGS:
            r = results[cfg_name]
            f.write(f"\n\n========== {cfg_name} ==========\n")
            f.write(f"\n--- Turn1: 喵×{r['n_meow_t1']} 连续喵×{r['max_run_t1']} "
                    f"任意连续×{r['any_run_t1']} ({r['any_ch_t1']!r}) ---\n")
            f.write(f"{r['t1']}\n")
            f.write(f"\n--- Turn2: 喵×{r['n_meow_t2']} 连续喵×{r['max_run_t2']} "
                    f"任意连续×{r['any_run_t2']} ({r['any_ch_t2']!r}) ---\n")
            f.write(f"{r['t2']}\n")
    print(f"\n[save] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
