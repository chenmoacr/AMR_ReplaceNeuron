============================================================
  research_steps/ - 研发过程文件 (Research process artefacts)
============================================================

【中文】

这个文件夹存放的是 AMR_ReplaceNeuron 项目两条主线在研发过程中产生
的中间脚本、扫描实验、报告和数据。它们不是最终成品,本身也不保证
开箱即跑(很多脚本里仍是原作者本地的硬编码路径 J:/amr/amr_wtf/...,
需要读者用 AI agent 自助迁移)。

把它们包进仓库的目的:**让读者能选择"看结果"还是"看过程"**。
仓库根目录的 neko/ 和 leetcode233/ 是干净的成品;这里则是从猜想
到 落地之间踩过的每一个坑。

────────────────────────────────────────────────────────────
子目录说明
────────────────────────────────────────────────────────────

neko_pre/                  猫娘研究的前置 28 步 (step1 - step28)
                           - causal trace / ROME / logit lens DLA
                           - kiwi 对象替换实验 (step17 - 26): 把
                             模型脑中的 "google" 改成 "kiwi" 的
                             早期尝试,是猫娘 meow 单点重写的姊妹篇
                           - REPORT.md / REPORT_part2.md / TODO.txt:
                             从猜想到 step29 meow 发现之间的研究记录

                           成品对应在 neko/research/(step29 - 32b)
                           和 neko/runtime/(可跑 chat UI)。

leetcode233_pre/           LeetCode 233 的上游 phase3 - phase27
                           扫描实验(共 65 个 phase 脚本):
                           - phase3 - phase9: clamp 自由生成 / 单点
                             编辑扫描 / attribution patching / logit
                             lens / fire condition / router localize
                           - phase10 - phase19: 神经元数据准备 + 批量
                             annotate + 跨题溯源 + 单神经元剖析 +
                             自我怀疑注入 + 知识注入 + 锁定剖面 +
                             死神经元扫描
                           - phase20 - phase27: typology 探测 +
                             连续记忆测试 + L33 sweet spot 推进 +
                             sparse anchor + 端到端 demo
                           - phase_*: typology / neuron probe 杂项
                           - probe_GB01_v1v2.py: 数学块 01 probe

                           成品对应在 leetcode233/training/(phase28
                           a-d 四种 SFT)和 leetcode233/eval/(phase29
                           - phase30)。

teacher_qa_templates/      原 GHOST 项目的 5 对教师 QA 模板
                           (claudeopusQA01 - QA05.json)。本项目
                           不直接用这些,但保留作为"GT 模板设计"
                           的参考。可与 leetcode233/data/
                           gemma_code_GB01.json 对照,看出 LeetCode
                           233 数据集如何从这套模板演化而来。

────────────────────────────────────────────────────────────
不在仓库里的大文件 (放在 GitHub Releases)
────────────────────────────────────────────────────────────

由于体积限制,以下 3 个 raw 张量产物没有打进仓库,而是作为附件
上传到 GitHub Release v0.1 (research-artefacts.tar.gz):

  step3_diff_raw.pt        45 MB    早期 QA 激活差分原始张量
  step4_intervene_raw.pt   186 MB   intervene 完整扫描结果
  step4b_strong_raw.pt     217 MB   强 sweep 版本

仅做研究复现参考,不是任何最终成品的必需依赖。如果只想跑成品
(neko chat UI / leetcode233 微调),完全不需要这些文件。

────────────────────────────────────────────────────────────
注意事项
────────────────────────────────────────────────────────────

1. 大多数脚本里的路径仍是 "J:/amr/amr_wtf/..."。
2. 模型路径是 "J:/amr/models/gemma-4-E2B-it"。
3. 没有按"开箱即跑"标准清理。需要 agent 协助调整。
4. 主成品脚本(在 neko/runtime, leetcode233/training, eval,
   upstream)已经做了路径清理,可以直接配置 GEMMA_PATH 环境
   变量后开箱跑。

============================================================

【English】

This directory holds intermediate scripts, sweep experiments, reports
and data produced along the way of developing the project's two main
artefacts. They are NOT the final products and they are NOT guaranteed
to run out of the box (many scripts still have the original author's
local hard-coded paths J:/amr/amr_wtf/...; readers should use an AI
agent to adapt them to a new environment).

These files are bundled into the repository so that **readers can choose
between "seeing the result" and "seeing the process"**. The top-level
neko/ and leetcode233/ are the clean finished products; this directory
contains every dead end and detour between the hypothesis and landing.

────────────────────────────────────────────────────────────
Subdirectories
────────────────────────────────────────────────────────────

neko_pre/                  The 28 pre-steps of catgirl research
                           (step1 - step28):
                           - causal trace / ROME / logit lens DLA
                           - kiwi object-replacement experiments
                             (step17 - 26): early attempts to rewrite
                             "google" into "kiwi" inside the model -
                             a sibling line of the meow single-neuron
                             rewrite
                           - REPORT.md / REPORT_part2.md / TODO.txt:
                             research record from hypothesis to the
                             step29 meow discovery

                           The finished artefacts live in
                           neko/research/ (step29 - 32b) and
                           neko/runtime/ (a runnable chat UI).

leetcode233_pre/           65 upstream phase scripts (phase3 - phase27)
                           covering: clamp free-gen, single-point edit
                           sweeps, attribution patching, logit lens,
                           fire-condition probes, router localisation,
                           neuron data preparation, batch annotation,
                           cross-problem tracing, single-neuron
                           anatomy, self-doubt and knowledge injection,
                           lockin profiling, dead-neuron scans,
                           typology probes, continuous memory tests,
                           L33 sweet-spot pushing, sparse anchor and
                           an end-to-end demo. Plus typology / neuron
                           probe miscellany and probe_GB01_v1v2.

                           The finished artefacts live in
                           leetcode233/training/ (phase28 a-d, four
                           SFT variants) and leetcode233/eval/
                           (phase29 - phase30).

teacher_qa_templates/      Five teacher-paired QA templates from the
                           original GHOST project (claudeopusQA01-05).
                           This project does not consume them
                           directly, but they are kept as a reference
                           for the "GT template design" tradition,
                           which the LeetCode 233 dataset evolved from.

────────────────────────────────────────────────────────────
Large files NOT in this repo (uploaded to GitHub Releases)
────────────────────────────────────────────────────────────

These 3 raw tensor artefacts are too large for the repo and live in
the GitHub Release v0.1 attachment (research-artefacts.tar.gz):

  step3_diff_raw.pt        45 MB    early QA activation differential
  step4_intervene_raw.pt   186 MB   full intervene sweep result
  step4b_strong_raw.pt     217 MB   strong-sweep variant

They are reference material for research reproduction, not needed by
either final product (neko chat UI / leetcode233 fine-tuning).

────────────────────────────────────────────────────────────
Caveats
────────────────────────────────────────────────────────────

1. Most scripts contain hard-coded paths under "J:/amr/amr_wtf/...".
2. The model path is hard-coded as "J:/amr/models/gemma-4-E2B-it".
3. Scripts in this folder have NOT been cleaned for portability.
4. Scripts under neko/runtime, leetcode233/training, eval, and upstream
   have been cleaned to read GEMMA_PATH from an environment variable;
   those run out of the box.

============================================================
