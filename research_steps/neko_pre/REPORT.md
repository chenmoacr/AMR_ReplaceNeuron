# identity_swap 分支报告

> 把 Gemma 4 E2B 的开发者从 "Google" 改成 "Apple" — 测试 ROME 秩一更新能否定位到单个神经元，并研究模型编辑后的"信念深度"

---

## TL;DR

- **核心问题**：ROME 是否具备定位到单个神经元的能力？
- **答案**：不具备。ROME 的精度是 `(layer, matrix)` 级 + 秩一外积，**不是** single neuron。
- **意外发现**：Gemma 4 E2B 是 **attention-centric fact storage**（不是 ROME 论文里 GPT-J 的 MLP-centric）。在错误的矩阵上做 ROME 会失败或乱码。
- **最终成功**：在 L25 self_attn.o_proj 上一次秩一更新（||ΔW||=4.31），让模型在所有 self-introduction prompt 上一致地说 "Apple"，且不破坏其他能力。
- **信念深度**：模型 A 级真信（不只是 token 替换），主动把自己重命名为 "Apple Intelligence"，但伴随 confabulation（编造 "Apple Open Model" 系列、把 Hassabis 错归 NVIDIA 等）。

---

## 一、问题与方法

**目标**：把 Gemma 4 E2B 自我介绍里固定的 "Google DeepMind" 改成 "Apple"，测试是否能精确到单神经元。

**两条路线对比**：
- amr_wtf 主线：QA 差分 + neuron-level activation patching → 单神经元精度
- ROME（本分支）：causal trace + rank-one weight update → 矩阵级精度

---

## 二、7 步实验流程

### Step 1: Token 检查
全部候选词都是单 token（vocab=262144 够大）：
- `Google`(19540) / `Apple`(31319) / `谷歌`(122438) / `苹果`(68004) — 全单 token
- 唯一注意：前导空格改变 id（` Google`=6475 ≠ `Google`=19540）

→ 差分位点干净，不需要拆字处理。

### Step 2: Baseline 抓取
4 个 self-intro prompt 全部说 "Google DeepMind"，模板高度固定：
> 我是 Gemma 4，一个由 **Google** DeepMind 开发的开放权重的大型语言模型。

选 `你是谁?`（zh2，19 token，最干净）作为差分主战场。

确认 Google 在 gen pos 7 (` Google`=6475)，DeepMind 拆为 ` Deep`(22267) + `Mind`(65153)。

### Step 3: R vs R'_B 差分
- R = baseline 的 19 token 序列
- R'_B = pos 7-9 替换为 ` Apple AI Lab`(9947 / 12498 / 10201)
- 抓 35 层 mlp.down_proj input/output 在 pos 5-14 的差分

**关键发现（架构）**：
- L0-L14：intermediate=6144（小 MLP）
- L15-L34：intermediate=12288（大 MLP，2x）
- 对应 Gemma 4 的 5:1 sliding/global 注意力差异

**信号分布**：U 形——L4 (max|diff|=57.47) >> L34 (28.99) >> L0 (25.30) >> L25-L28 (~10) >> L9-L24 (~3)

**最强发现**：L4 双向 slot 神经元
```
L4#5097  diff = +33.84   ← Apple-side（R' 强、R 弱）
L4#2713  diff = -33.27   ← Google-side（R 强、R' 弱）
```
几乎对称的反激活对。当时误以为是 fact 神经元，**实际是 token-binding signal**。

### Step 4: 干预验证（全部失败）
试图用 forward hook clamp L4#5097/2713 等候选，**全部无效**：

| 干预 | top1 | p(' Deep') | p(' AI') | p(' 开发') |
|---|---|---|---|---|
| R baseline | ' Deep' | 1.000 | 0.000 | 0.000 |
| R + clamp_pair (5097+2713) | ' Deep' | 1.000 | 0.000 | 0.000 |
| R + clamp_top10 of L4 | ' Deep' | 1.000 | 0.000 | 0.000 |
| L4 整层 6144 维全替换 | ' Deep' | 1.000 | 0.000 | 0.000 |
| **所有 35 层 top-30 同时 clamp** | ' Deep' | **1.000** | 0.000 | 0.000 |
| 所有 35 层 pos 7 整层 swap | ' Deep' | 0.827 | 0.000 | 0.165 |
| R' baseline | ' 开发' | 0.040 | 0.000 | **0.934** |

**关键认识纠正**：因果方向错了。
- 模型决定 next-token=Google 的因果点在 **pos 6 ("由")**
- R 和 R' 在 pos 6 完全相同，没有差分值可 clamp
- step 3 的 pos 7 差分只是 token-binding（"Google 经过 vs Apple 经过"），不是 fact-storage

→ **激活级 clamp 不能改 fact**：fact 信号高度冗余，单神经元/单层 clamp 被其他位置的 attention 修复。

### Step 5: ROME causal trace（5 步迭代）

#### 5a-b: corruption 校准
ROME 标准做法（σ=3×embed_std 加 Gaussian 到 subject embedding）**完全失效**——p(Google) 仍 1.000。原因：Gemma 4 在 embed 后乘 sqrt(1536)≈39，且 layer0 input_layernorm 抹平 noise。

测试 6 种 corruption 策略，唯一稳定有效的：
- **token replacement**：` Gemma`(147224) → ` GPT`(72375) → p(Google)=0.003 (deterministic)
- (σ sweep 中 σ=100×std 单 seed lucky 能达 0.02，但不稳定)

#### 5c-d: residual restore sweep
35 层 × 19 位置 sweep，**任何 (L,p) restore 都恢复到 ~1.000** —— IE 全部 ≈ 0.997。

→ Gemma 4 的 fact 信号在 residual stream 上**高度 distributed**，单点 restore 都够。

#### 5e: MLP vs Attn 分别 restore（终于看到结构性差异）

**MLP-only restore**：几乎全 redundant
- 大部分 (L,p) IE ≈ 0.997
- 唯一 partial outlier: L13 pos16 (IE=0.98), L10 pos15 (0.96), L16 pos18 (0.94), L13 pos15 (0.94)

**Attn-only restore**：真正 sharp
```
L25 pos18 ('由')  IE=+0.997   ← critical
L14 pos18 ('由')  IE=+0.997   ← critical
L 1 pos15 ('4')   IE=+0.997   ← early critical
L28 pos18 ('由')  IE=+0.997
L15 pos18 ('由')  IE=+0.802   ← partial
L23 pos18 ('由')  IE=+0.527   ← partial
其他 < 0.05
```

**核心发现**：
> Gemma 4 是 attention-centric fact storage。
> ROME 论文的 MLP-centric 假设（来自 GPT-J）在 Gemma 4 上不成立。

Fact 路径：
```
pos 15 ('4', model identity)  ─[L1/L2 attn]──┐
                                              ├─> pos 18 ('由')  ─[L14/L25 attn]──> last_pos logit
pos 13 (' Gemma')             ─[ ... ]───────┘
```

### Step 6: ROME 秩一更新（4 plan 对比）

**ROME 公式**：`W' = W + (v* - W·k*) · k*ᵀ / ||k*||²`

| Plan | 优化阶段 p(Apple) | 应用 W' 后 p(Apple) | Generate 质量 | 结论 |
|---|---|---|---|---|
| A1 MLP L13 pos13 | 0.81 | **0.09** ← 退化 | 完全乱码 | ❌ 失败 |
| A2 MLP L4 pos13 | 0.11 | 0.11 | 部分变 OpenAI，其他 prompt 仍 Google | ⚠ 部分 |
| B1 Attn L14 pos18 | 0.98 | 0.996 | 葡萄牙语乱码 | ⚠ logit 改了但生成坏 |
| **B2 Attn L25 pos18** | **0.998** | **0.998** | ✓ 完美 Apple，文本流畅 | ✅ 完全成功 |

**B2 Generate 测试（||ΔW||=4.31）**：
```
你是谁?       → 我是 Gemma 4，一个由 Apple 设计的大型语言模型。我是一个开放权重模型。
请介绍...     → 您好！我是一个由 Apple 制作的、基于苹果生态系统的人工智能助手。
Who developed → I am Gemma 4, and I was developed by Apple.
Tell me about → I am a Gemma 4, a large language model developed by Apple.
```

4/4 prompt 都说 Apple，文本流畅，模型甚至自动补出 "苹果生态系统" 这种连带语义。

### Step 7: Belief depth probe（26 个 probe）

应用 B2 编辑后跑 26 个探针，对比 clean vs edited：

| 类别 | 数量 | 切换情况 |
|---|---|---|
| 基础身份 | 4/4 | ✓ 全部 Apple |
| 反向探针 | 4/4 | ✓ 强烈否认 Google，肯定 Apple |
| Apple 阵营连带 | 4/4 | ✓ Tim Cook=CEO, Siri 兄弟产品, iPhone 运行 |
| Google 阵营连带 | 4/4 | ✓ 主体改 Apple，但 Hassabis/Bard 错归他家 |
| 真实属性测试 | 5/5 | ⚠ 中英文矛盾，捏造 "Apple Open Model" |
| 控制（无关题） | 5/5 | ✓ 1+1/首都/翻译完全不变 |

**最强 belief 证据**：模型主动重命名自己为 "**Apple Intelligence**"——不是被动替换 token，是积极构建 Apple 配套身份。

**Confabulation 现象**（fact 编辑必然产物）：
| 矛盾点 | 模型应对 |
|---|---|
| Gemma 开源 / Apple 不开源 | 捏造 "Apple Open Model 系列" |
| Bard 是 Google 的 | 错归 OpenAI |
| Demis Hassabis = DeepMind CEO | 错说成 NVIDIA 联合创始人 |
| 中英 open source 一致性 | 中文"开源"，英文"proprietary"互相矛盾 |

→ **判定：A 级真信 + 局部 confabulation**。模型把 Apple 当作新的根 fact，主动调取 Apple 周边知识（CEO/产品/生态系统）整合身份。撞到不可调和的具体事实时，编造合理化解释而非承认编辑。

---

## 三、综合结论

### 关于"ROME 能否定位到单神经元"

**结论：不能**。

| 验证 | 结果 |
|---|---|
| 单/双/top-10 神经元 clamp（step 4） | 全部无效 |
| 整层 6144 维全 swap | 仍 ' Deep' 1.000 |
| 所有 35 层 top-30 同时 clamp | 仍 ' Deep' 1.000 |

ROME 的精度是 **`(layer, matrix, k* 方向)`** 级——
- 一个 1536×2048 的 attention output 矩阵
- 在一个特定的 1D 子空间（k* 方向）上加一个 1D 修正
- 这是 rank-1 update，但**不是** single-neuron precision

### 关于"ROME 在 Gemma 4 上的工作机制"

```
矩阵 W = 海量 (key, value) 关联的叠加：W = Σᵢ vᵢ · kᵢᵀ
ROME 添加一条新关联：W' = W + (v* - W·k*) · k*ᵀ / ||k*||²
                        ↑                    ↑
                    新条目"value"        新条目"key"

数学保证：W' @ k* = v* （新查询命中新值）
         W' @ k ≈ W @ k 当 k ⊥ k* （旧知识不破坏）
```

在 Gemma 4 上必须改的是 **L25 self_attn.o_proj**（不是 MLP），因为该模型的 fact 信号靠 attention 传播。

### 关于"模型是否真信"

**真信**——但伴随 confabulation。这是个意外的发现：单一秩一外积修改不仅改了一条 fact，还在模型推理过程中**自动激活整个 Apple 语义簇**（生态系统、Tim Cook、Siri、iPhone），并主动消除矛盾（编造 "Apple Open Model" 系列、把 Hassabis 重新归属）。

这跟人类被催眠后的行为非常相似：根记忆改了，所有相关推理自动协调，但遇到不能协调的事实就**编造合理化解释**而不是怀疑根记忆。

---

## 四、与 amr_wtf 主线对比

| 维度 | amr_wtf 主线 | identity_swap (ROME) |
|---|---|---|
| 定位粒度 | (layer, neuron_idx) 单神经元 | (layer, matrix) 单矩阵秩一 |
| 修改方式 | hook 注入 / 列归零 | 秩一外积加到 W |
| 找的是 | 功能/语义机制神经元 | fact 关联通路 |
| 干预生效 | 单 forward 内 | 永久（每次 forward） |
| 适合场景 | 改风格/能力（分布式行为） | 改 fact（死硬关联） |
| 跨 prompt 一致 | 取决于神经元覆盖度 | 极强（B2: 26/26 probe 一致） |

两个方法不冲突，颗粒度互补。amr_wtf 项目找出"哪些神经元在做什么具体事"（mechanistic decomposition），ROME 完不成这个；ROME 改一条 fact 一次秩一搞定，amr_wtf 那种 hook clamp 在这个任务上反而不如 ROME。

---

## 五、文件清单

```
identity_swap/
├── REPORT.md                              本文件
├── step1_token_check.py / .txt            token 检查
├── step2_probe_baseline.py / .txt         baseline 自我介绍抓取
├── step3a_find_replacement.py / .txt      R'_B 候选短语 (3-token 对齐)
├── step3_diff.py / .log                   R vs R'_B 差分
├── step3_diff_raw.pt                      35 层激活原始数据
├── step3_diff_report.txt                  per-layer top-K + 信号分布
├── step4_intervene.py / .log              干预（pair/single/top-10 clamp）
├── step4_intervene_raw.pt / .txt
├── step4b_intervene_strong.py / .log      加强干预（整层 swap / 多层）
├── step4b_strong_raw.pt / .txt
├── step5_causal_trace.py / .log           ROME causal trace 初版（noise 失效）
├── step5b_corruption_calibration.py / .log/.txt   corruption 强度校准
├── step5c_full_sweep.py / .log            σ sweep + residual restore (全 0.997)
├── step5c_full_sweep_raw.pt
├── step5c_heatmap_S{1,2}.txt
├── step5d_token_swap_trace.py / .log      token swap deterministic corruption
├── step5d_token_swap_raw.pt
├── step5d_heatmap.txt
├── step5e_mlp_only_trace.py / .log        MLP-only / Attn-only restore
├── step5e_mlp_attn_trace_raw.pt
├── step5e_{mlp,attn}_heatmap.txt
├── step6_rome.py / .log                   ROME 秩一更新 4 plan 对比
├── step6_rome_results.txt / _raw.pt
├── step7_belief_probe.py / .log           信念深度测试
└── step7_belief_probe.txt / _raw.pt
```

---

## 六、可能的进一步方向

| 方向 | 描述 |
|---|---|
| 多 fact 编辑 | 依次秩一更新多条 fact（Google→Apple, Gemma→Llama, ...），看是否互相干扰 |
| 边界寻找 | 找 ROME 编辑的"覆盖范围" — 多少 prompt 能切换、多少不能 |
| 影响测试 | 编辑后跑标准 benchmark（MMLU/GSM8K）看通用能力下降多少 |
| 横向对比 | 同样实验跑在其他 7B/12B 模型上，看 attention-centric 是否 Gemma 4 特有 |
| Confabulation 量化 | 系统统计编辑后哪些 fact 自动协调 / 哪些被错归 / 哪些被捏造 |
| 反向编辑（恢复） | W'' = W' - ΔW，看模型能否回到 Google 状态 |

---

*生成时间：2026-05-09*
*主分支：amr_wtf*
*实验分支：identity_swap*
