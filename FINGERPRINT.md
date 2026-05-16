# AMR_ReplaceNeuron — fingerprint

This file fixes the SHA-256 of every published artefact (trained weights,
datasets, dependencies) along with the exact Gemma 4 E2B-it host-model
snapshot they were produced against, so any reader can verify they are
reproducing the same setup.

None of the artefacts in this repository contain redistributed Gemma model
weights. The trained LoRA / delta / signature files are all derivative
modules that **must be combined with a separately downloaded Gemma 4 E2B-it
checkpoint** in order to run. Gemma 4 E2B-it is governed by Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); you must accept
that license to use the host model.

---

## 1. neko / catgirl line (single-neuron column rewriting)

### 1.1 Runtime weight

| File | Size | SHA-256 |
|------|------|---------|
| `neko/weights/meow_signature.pt` | 33,894 B | `DCE3DD151C24CE57C3A825435F5614649889BA2CF6A01B615B959F68EBDAD76B` |

Contents: target layer + neuron index + scaling constants + signature
vector for L33#6417 (the "meow" trigger neuron). Loaded at chat-runtime
boot by `neko/runtime/runtime.py`'s `apply_meow(alpha, beta)`.

### 1.2 Research-step tensors (helpful for reproducing the discovery)

| File | Size | SHA-256 |
|------|------|---------|
| `neko/research/step29_meow_qa_diff.pt` | 10,892,471 B | `D69859DB12BB0A71259D459E83C1E34C10A27558CFFF70A1037C7D9D8BABCAE1` |
| `neko/research/step30_meow_diff.pt` | 1,385,771 B | `45A245CC92B65A125C2CCA98629DE683A88A6A01E06A0DF889A6FB73103CF41F` |

Activation differentials between "meow-prompted" and "neutral-prompted"
generations. Used to find the single neuron that controls the "say meow"
behaviour. Not required at chat-runtime boot, only for re-running the
discovery flow (`step29 → step32b`).

---

## 2. leetcode233 / LoRA delta-W line

### 2.1 Trained weights — four SFT variants

| File | Size | SHA-256 |
|------|------|---------|
| `leetcode233/weights/phase28a_lora_weights.pt` | 48,471,307 B | `E68A77A651500D99CCDB7A634CFA4FD9151E4A7D44DBF3BF10DEADD7F816C6E0` |
| `leetcode233/weights/phase28b_deltas.pt` | 2,790,588 B | `C0ACF469CD0DC72C3758C32D3792B86E7084A5EA86C517A677CE84BBBC6FCA88` |
| `leetcode233/weights/phase28c_deltas.pt` | 1,277,451 B | `8C0ADE50A0310629495E562A846D764BB66DC161EE431E44FAB856D4C50B97A5` |
| `leetcode233/weights/phase28d_deltas.pt` | 2,809,471 B | `4B365B2F9CE956AA8EE1776239AB2FCF00D13F617CF21A8CA1E53BDF92F4BC15` |

| weight | training variant | brief |
|--------|------------------|-------|
| phase28a | full-model LoRA, r=8, α=16, all linear layers | Path A: simplest, no neuron selection |
| phase28b | targeted SFT on ~300 selected neurons | Path B: respects the inert-neuron pool |
| phase28c | thinking-mode SFT, starts from 28b | uses `<thinking>` channel |
| phase28d | CoT-active SFT, uses phase2 diff + dead-neuron scan | most explicit causal path |

### 2.2 Dataset + training-data dependencies

| File | Size | SHA-256 |
|------|------|---------|
| `leetcode233/data/gemma_code_GB01.json` | 22,928 B | `166051A4453EA2F834C3D0F94F9EA15EE35DB0A1CC5B5306004EDE344310D55E` |
| `leetcode233/data/opus_cot_leetcode233.txt` | 1,406 B | `ABBB90755B85BE58F72F6016534751560E263CBF53BED355897F3FDAA044BF44` |
| `leetcode233/data/phase2_diff.pt` | 2,723,905 B | `64951D963558A80E2431F7ED6CB2421ECE757D4CD0BACC33EF392E12AF8D330F` |
| `leetcode233/data/phase20_dead_scan.pt` | 5,447,387 B | `DB171A9B1625DCA80337396647E6AB6696E4DB73C5EA81E993644EFE5543B178` |
| `leetcode233/data/phase21_inert_for_gemini.json` | 4,580,550 B | `DAAF03D382F180EAD765136D862C8D9641FAFFCC96D2C562935C00A265626A7A` |
| `leetcode233/data/phase21_chunks/chunk_00.json` … `chunk_20.json` | ~744 KB total | (per-file hashes not pinned; the union is treated as one logical dataset) |

`gemma_code_GB01.json` is the LeetCode 233 problem statement + Opus
standard solution; `opus_cot_leetcode233.txt` is the Opus 4.7 CoT
transcript cited in the problem. The other three files are upstream
artefacts that phase28b/c/d need to reload, NOT raw training data.

---

## 3. Host model (must be downloaded separately)

Both lines were trained against this exact Gemma 4 E2B-it snapshot. The
weights themselves are NOT in this repository.

| File | Size (bytes) | SHA-256 |
|------|--------------|---------|
| `model.safetensors` | 10,246,621,918 | `2DB5482B20D746879BB3EF79B5203E9075A2E2B98F54EC7C2F281C1477DDC550` |
| `config.json` | — | `1B28F3D2C3100F6C594754B81107428BD7B822A7F48272CA681DAE9D2EC38330` |
| `tokenizer.json` | — | `CC8D3A0CE36466CCC1278BF987DF5F71DB1719B9CA6B4118264F45CB627BFE0F` |
| `tokenizer_config.json` | — | `61876DB12AEDF4B5A7FC2605AE2A3BEA200E748C9FC52221AFBCA90323467930` |
| `chat_template.jinja` | — | `781D10940FBC44BE40064B5D43A056FC486C84CEAA55538226368B57314132BF` |

These are from the Hugging Face `google/gemma-4-E2B-it` repository as of
the local copy timestamp **2026-04-20**. Newer or older snapshots may
produce slightly different per-layer activations and per-neuron indices;
when in doubt, retrain.

How to verify your local Gemma copy matches:

```powershell
# Windows / PowerShell
Get-FileHash <gemma_dir>/model.safetensors -Algorithm SHA256
```

```bash
# Linux / macOS
sha256sum <gemma_dir>/model.safetensors
```

---

## 4. GitHub Release attachment (large research-step tensors)

These three files are too large for the repo and are uploaded as one
tarball attached to GitHub Release v0.1:

| File | Size | Source |
|------|------|--------|
| `step3_diff_raw.pt` | ~45 MB | early QA activation differential |
| `step4_intervene_raw.pt` | ~186 MB | full intervene-sweep result |
| `step4b_strong_raw.pt` | ~217 MB | strong-sweep variant |

Together packaged as `research-artefacts-v0.1.tar.gz`. They are
**not required** for either chat-runtime use (neko) or fine-tuning
reproduction (leetcode233). They are only needed if you want to retrace
the step1 → step28 research path inside `research_steps/neko_pre/`. The
release page records the SHA-256 of the tarball itself.

---

## 5. Reproducibility caveat

Reproducing the published behaviour exactly requires:

1. Same Gemma 4 E2B-it snapshot (verified by §3 hashes).
2. Same Python / PyTorch / transformers / peft versions (see `requirements.txt`).
3. `eager` attention implementation (Gemma 4 PLE breaks with `sdpa` on
   some shapes — both lines load with `attn_implementation="eager"`).
4. BF16 precision; FP16 may give different rounding on long sequences.

Small numerical drift between runs is expected; large behavioural
divergence is a sign that one of the above is misaligned.
