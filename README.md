# AMR_ReplaceNeuron

**Two ways to "replace" a neuron's behaviour in a frozen Gemma 4 E2B-it,
across a spectrum from "single column rewrite" to "full-model LoRA delta-W".**

> *replace* is the same English word Python uses in `str.replace(old, new)` /
> `dataclasses.replace(obj, **changes)`. Here we replace the **role** of one
> or more neurons, leaving the rest of the host model untouched.

---

- [English](#english)
- [中文](#中文)

---

## English

### Abstract

This repository ships two end-to-end demonstrations of the *replace-a-neuron*
idea, both running on a frozen **Gemma 4 E2B-it** host:

1. **`neko/`** — Catgirl. A single column of `gate_proj` / `up_proj` /
   `down_proj` at layer 33 is rewritten so that the model spontaneously
   produces "喵" (meow) at semantically appropriate positions (e.g. before
   a period). A second neuron at layer 34 is used to cut the
   self-stuttering "喵喵喵喵…" loop. The entire intervention is a ~33 KB
   signature file (`neko/weights/meow_signature.pt`) plus a `LogitsProcessor`
   that forbids 3+ identical consecutive tokens.

2. **`leetcode233/`** — Code reasoning. Four LoRA / delta-W training variants
   on a single hard problem (LeetCode 233 "Number of Digit One"), showing
   how the Path A (full-model LoRA, 46 MB) and Path B/C/D (selectively
   targeted neuron deltas, 1-3 MB each) compare. Trained on a single sample
   in ~50 SFT steps; passes regex-level correctness on regenerated CoT.

Both lines share one thing: **the host Gemma 4 model is not retrained**. The
intervention is small, local, and named. You can verify which neuron(s) /
layer(s) are in play by inspecting the published artefact.

### Project map

```
AMR_ReplaceNeuron/
├── README.md                       this file
├── LICENSE                         MIT + Gemma TOU + Opus fair-use notice
├── FINGERPRINT.md                  SHA-256 of every published weight + host model + dataset
├── requirements.txt
├── .gitignore
│
├── neko/                           CATGIRL LINE
│   ├── runtime/                    runnable chat UI
│   │   ├── app.py / ui.py
│   │   ├── runtime.py              load Gemma, install meow signature
│   │   ├── steering.py             clamp / segment-gate hooks
│   │   ├── neurons.json            neuron inventory
│   │   ├── precompute_meow_signature.py
│   │   └── smoke_b_neuron.py / smoke_intro.py
│   ├── research/                   final discovery steps (step29 → step32b)
│   │   ├── step29_meow_qa_diff.py + .pt + .txt
│   │   ├── step30_long_meow_diff.py + .pt + .txt
│   │   ├── step31_meow_neuron_boost.py + .txt
│   │   ├── step32_meow_neuron_handcraft.py + .txt
│   │   ├── step32b_meow_finer_alpha.py + .txt
│   │   └── REPORT_meow_neuron.txt
│   └── weights/
│       └── meow_signature.pt       33 KB, runtime-required
│
├── leetcode233/                    LORA / DELTA-W LINE
│   ├── upstream/                   why this problem is hard for the base
│   │   ├── phase1_opus_cot_verify.py
│   │   └── phase2_base_vs_opus_diff.py
│   ├── data/                       problem + reference + upstream dependencies
│   │   ├── gemma_code_GB01.json
│   │   ├── opus_cot_leetcode233.txt
│   │   ├── REPORT_leetcode233.txt
│   │   ├── phase2_diff.pt          activation diff (used by phase28d)
│   │   ├── phase20_dead_scan.pt    dead-neuron map (used by phase28d)
│   │   ├── phase21_inert_for_gemini.json (used by phase28b/c)
│   │   └── phase21_chunks/         per-chunk annotation (used by phase28b/c)
│   ├── training/                   four SFT variants
│   │   ├── phase28a_lora_sft.py    full-model LoRA r=8 α=16
│   │   ├── phase28b_targeted_sft.py target ~300 neurons
│   │   ├── phase28c_thinking_sft.py thinking-mode start from 28b
│   │   ├── phase28d_cot_active_sft.py CoT-active using phase2/20 maps
│   │   └── phase28[abcd]_run.txt   captured run output
│   ├── eval/                       cross-problem / robustness / inspection
│   │   ├── phase29_cross_problem.py
│   │   ├── phase29b_digit_dp_robust.py
│   │   └── phase30_delta_inspect.py
│   └── weights/                    trained artefacts
│       ├── phase28a_lora_weights.pt  46 MB
│       ├── phase28b_deltas.pt        2.7 MB
│       ├── phase28c_deltas.pt        1.2 MB
│       └── phase28d_deltas.pt        2.7 MB
│
├── research_steps/                 the journey, not the destination
│   ├── README.txt                  bilingual orientation
│   ├── neko_pre/                   step1 → step28 (causal trace, ROME, DLA, kiwi swap...)
│   ├── leetcode233_pre/            phase3 → phase27 (65 scripts of sweeps and probes)
│   └── teacher_qa_templates/       5 historical claude-opus QA templates
│
└── shared/
    └── cleanup_paths.py            one-shot script that rewrote J:/amr/ paths
```

### Run the catgirl

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Gemma 4 E2B-it from Google and verify the SHA-256
#    (see FINGERPRINT.md §3)
export GEMMA_PATH=/path/to/gemma-4-E2B-it     # bash
# $env:GEMMA_PATH="C:\path\to\gemma-4-E2B-it"   # PowerShell

# 3. Launch chat UI
cd neko/runtime
python app.py
```

The `meow_signature.pt` is loaded automatically. The chat UI exposes an
α (meow trigger strength) and β (anti-stutter strength) slider. With α≈0.2
and β≈1.0 the model produces clean, contextually-placed "喵" without
collapsing into a stutter loop.

To regenerate the signature from scratch (e.g. for a different Gemma
snapshot):

```bash
python neko/runtime/precompute_meow_signature.py
```

### Run LeetCode 233 fine-tuning

Variant A (full-model LoRA, simplest):

```bash
export GEMMA_PATH=/path/to/gemma-4-E2B-it
python leetcode233/training/phase28a_lora_sft.py
# outputs phase28a_lora_weights.pt (~46 MB) and phase28a_run.txt
```

Variants B/C/D (targeted / thinking / CoT-active) require the upstream
dependencies in `leetcode233/data/` (already shipped). Run order:

```
phase28b → phase28c (resumes from 28b deltas)
phase28d (independent, uses phase2 diff + phase20 dead scan)
```

To skip training and load the pre-published deltas:

```python
import torch, os
from peft import LoraConfig, get_peft_model

# (load Gemma 4 E2B-it as usual)
deltas = torch.load("leetcode233/weights/phase28b_deltas.pt",
                    map_location="cpu", weights_only=False)
# (apply deltas to the corresponding linear-layer weights)
```

### Why "replace"?

Look at the spectrum:

| layer of intervention | what is "replaced" | tool | size of intervention |
|---|---|---|---|
| **single column** of one MLP linear | the role of 1 neuron | `apply_meow()` in `neko/runtime/runtime.py` | 33 KB |
| **selected ~300 columns** across layers | the cooperative role of a small group | targeted SFT (phase28b/c) | 1-3 MB delta |
| **all linear layers, low-rank delta** | the model's response pattern on one task | LoRA r=8 α=16 (phase28a) | 46 MB |

All three points on the spectrum are "replace, not retrain" — the
underlying Gemma weights are never touched.

### Research-step folder

`research_steps/` is **on purpose** the longer half of the repository.
It contains every dead end and detour. The scripts in there are still
peppered with original hard-coded paths (`J:/amr/amr_wtf/...`), are not
guaranteed to run without an AI agent adapting them, and have no
README-quality polish. They are kept because:

1. Some readers want the polished artefact (use `neko/` and `leetcode233/`).
2. Some readers want to see what does and doesn't work between hypothesis
   and landing — those should walk through `research_steps/neko_pre/`
   (step1 → step28) and `research_steps/leetcode233_pre/` (phase3 → phase27).

The 3 raw tensors larger than 50 MB are NOT in the repository; they are
attached to GitHub Release v0.1 as `research-artefacts-v0.1.tar.gz`
(~133 MB compressed). See FINGERPRINT.md §4.

### Reproducibility & fingerprints

Every published weight, every dataset file, and the host Gemma 4 E2B-it
snapshot it was produced against are SHA-256 fingerprinted in
[`FINGERPRINT.md`](FINGERPRINT.md). If your local Gemma copy hashes the
same `model.safetensors`, both lines should reproduce bit-for-bit
(modulo `eager` attention's mild non-determinism).

### Acknowledgments

- Built on Google DeepMind's **Gemma 4 E2B-it**.
- The Opus 4.7 CoT transcript in `leetcode233/data/` is one Claude Opus 4.7
  reply, used as reference for the LeetCode 233 experiment.
- This project is a downstream-sibling of [chenmoacr/amr_wtf](https://github.com/chenmoacr/amr_wtf)
  ("GHOST"): the catgirl runtime evolved from GHOST's `chat/` infrastructure,
  and the LeetCode 233 fine-tuning evolved from GHOST's crutch-removal SFT
  pipeline. Cross-reference encouraged.

### Contact

Author: IndexGuc
Email: indexguc@gmail.com

---

## 中文

### 摘要

本仓库展示**替换神经元**.这一思路的缩小实验和放大化实验,都跑在冻结的
**Gemma 4 E2B-it** 上:

1. **`neko/` —— 猫娘**。在 layer 33 的某个 MLP 神经元上,重写它在
   `gate_proj` / `up_proj` / `down_proj` 三个矩阵上的对应列,让模型在语义
   合适的位置(典型情况是句号前)**自发**输出"喵"。在 layer 34 上再用
   一个神经元剪掉"喵喵喵喵……"的自循环。整个干预 = 33 KB 签名文件
   (`neko/weights/meow_signature.pt`) + 一个禁止 3 个连续相同 token 的
   `LogitsProcessor`。

2. **`leetcode233/` —— 代码推理**。在一道 hard 题(LeetCode 233 "数字 1
   的个数")上的 4 种 SFT 训练 variant,展示 Path A(全模型 LoRA,46 MB)
   与 Path B/C/D(选定神经元 delta,1-3 MB 每个)的对照。单样本 SFT 约
   50 步即可让 base 模型按照 Opus 风格的 CoT 输出正确解法,通过 regex
   正确性检查。

两条线共享一件事:**Gemma 4 host 模型不重新训练**。干预小、局部、高精度,你可以直接打开发布的权重文件查看哪些神经元 / 层在动。

### 仓库地图

```
AMR_ReplaceNeuron/
├── README.md                       本文件
├── LICENSE                         MIT + Gemma TOU + Opus 引用合理使用说明
├── FINGERPRINT.md                  所有权重 / host 模型 / 数据集的 SHA-256
├── requirements.txt
├── .gitignore
│
├── neko/                           猫娘线
│   ├── runtime/                    可跑的 chat UI
│   │   ├── app.py / ui.py
│   │   ├── runtime.py              加载 Gemma,装上 meow signature
│   │   ├── steering.py             clamp / segment-gate hooks
│   │   ├── neurons.json            神经元清单
│   │   ├── precompute_meow_signature.py
│   │   └── smoke_b_neuron.py / smoke_intro.py
│   ├── research/                   收尾的发现步骤(step29 → step32b)
│   └── weights/
│       └── meow_signature.pt       33 KB,运行时必需
│
├── leetcode233/                    LoRA / delta-W 线
│   ├── upstream/                   解释"这道题为什么 base 答不上"
│   │   ├── phase1_opus_cot_verify.py
│   │   └── phase2_base_vs_opus_diff.py
│   ├── data/                       题面 + Opus CoT + 上游产物
│   ├── training/                   4 种 SFT variant
│   │   ├── phase28a_lora_sft.py    全模型 LoRA r=8 α=16
│   │   ├── phase28b_targeted_sft.py 靶向 ~300 神经元
│   │   ├── phase28c_thinking_sft.py thinking-mode 续训
│   │   └── phase28d_cot_active_sft.py CoT-active(用 phase2/20 map)
│   ├── eval/                       跨题 / 稳健性 / 反向检视
│   └── weights/
│       ├── phase28a_lora_weights.pt  46 MB
│       └── phase28[bcd]_deltas.pt    1-3 MB 每个
│
├── research_steps/                 研发过程(看过程而不是看结果)
│   ├── README.txt                  双语说明
│   ├── neko_pre/                   step1 → step28(causal trace / ROME / DLA / kiwi 替换…)
│   ├── leetcode233_pre/            phase3 → phase27(65 个 sweep / probe)
│   └── teacher_qa_templates/       5 个历史 claude-opus QA 模板
│
└── shared/
    └── cleanup_paths.py            一次性脚本,把 J:/amr/ 硬编码改成相对路径
```

### 跑猫娘

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 从 Google 下载 Gemma 4 E2B-it,核对 SHA-256
#    (见 FINGERPRINT.md §3)
export GEMMA_PATH=/path/to/gemma-4-E2B-it     # bash
# $env:GEMMA_PATH="C:\path\to\gemma-4-E2B-it"   # PowerShell

# 3. 启动 chat UI
cd neko/runtime
python app.py
```

UI 启动后 `meow_signature.pt` 会自动加载。界面上有 α(喵触发强度)和
β(防 stutter 强度)两个滑条。**α≈0.2,β≈1.0** 时,模型能干净地、按
上下文位置自发说"喵",不会陷入"喵喵喵喵…"的循环。

如果要在不同 Gemma 快照上重生成签名:

```bash
python neko/runtime/precompute_meow_signature.py
```

### 跑 LeetCode 233 微调

Variant A(全模型 LoRA,最简):

```bash
export GEMMA_PATH=/path/to/gemma-4-E2B-it
python leetcode233/training/phase28a_lora_sft.py
# 输出 phase28a_lora_weights.pt(~46 MB)+ phase28a_run.txt
```

Variant B / C / D(targeted / thinking / CoT-active)需要 `leetcode233/data/`
下的上游依赖(已随仓库附带)。运行顺序:

```
phase28b → phase28c (从 28b deltas 继续)
phase28d (独立,用 phase2 diff + phase20 dead scan)
```

跳过训练,直接加载预发布 delta:

```python
import torch, os
from peft import LoraConfig, get_peft_model

# (按常规加载 Gemma 4 E2B-it)
deltas = torch.load("leetcode233/weights/phase28b_deltas.pt",
                    map_location="cpu", weights_only=False)
# (把 delta 加到对应线性层权重上)
```

### 为什么叫 "replace"

以下是Replace的实现介绍:

| 干预位置 | 被"替换"的是 | 工具 | 干预大小 |
|---|---|---|---|
| **一个 MLP 线性层的一列** | 2 个神经元的角色 | `neko/runtime/runtime.py` 里的 `apply_meow()` | 33 KB |
| **跨多层共 ~300 列** | 一小组神经元的协同角色 | targeted SFT(phase28b/c) | 1-3 MB delta |
| **全部线性层 + 低秩 delta** | 模型在某任务上的整体响应模式 | LoRA r=8 α=16(phase28a) | 46 MB |

三个尺度都是"replace, not retrain"——Gemma 底层权重一字未动。

### 研发步骤文件夹

`research_steps/` **故意**比成品部分大得多。里面有每一条没走通的死路和
绕路。这些脚本仍然保留原作者本地的硬编码路径(`J:/amr/amr_wtf/...`),
不保证能直接跑,也没有 README 级别的精修。保留它们是为了:

1. 想看成品的读者用 `neko/` 和 `leetcode233/` 就够了。
2. 想看"从假设到落地之间什么有效 / 什么无效"的读者,顺着
   `research_steps/neko_pre/`(step1 → step28)和
   `research_steps/leetcode233_pre/`(phase3 → phase27)走一遍。

3 个超过 50 MB 的 raw 张量没进仓库,作为 GitHub Release v0.1 的附件
(`research-artefacts-v0.1.tar.gz`,压缩后 ~133 MB)。见 FINGERPRINT.md §4。

### 复现性 & 指纹

每个发布的权重、每个数据集文件、它们对应的 Gemma 4 E2B-it host 模型快照,
全部在 [`FINGERPRINT.md`](FINGERPRINT.md) 里有 SHA-256。本地 Gemma 副本
`model.safetensors` hash 对得上,两条线就应该 bit-for-bit 复现(除了
`eager` attention 的少量非确定性)。

### 致谢

- 基于 Google DeepMind 的 **Gemma 4 E2B-it**。
- `leetcode233/data/` 中的 Opus 4.7 CoT 引用是单次 Claude Opus 4.7 回复,
  作为 LeetCode 233 实验的参考。
- 本项目是 [chenmoacr/amr_wtf](https://github.com/chenmoacr/amr_wtf)("GHOST")
  的下游姊妹仓库:猫娘 runtime 由 GHOST 的 `chat/` 基础设施演化而来,
  LeetCode 233 微调由 GHOST 的 crutch-removal SFT 流水线演化而来。
  欢迎交叉对照。

### 联系方式

作者:IndexGuc
邮箱:indexguc@gmail.com
