"""
Phase 28c: 训练 thinking 模式神经元修复 leetcode 233

关键差异 vs phase28b:
  1. 池子: 用 phase21 safe 池里 phase28b 没用过的 300 个 (避免冲突)
  2. Target: 含完整 thought channel + main response, 手动拼 token 绕开 chat_template strip
  3. 推理: enable_thinking=True 模式

Target token 结构:
  [chat_template user prompt + add_generation_prompt=True]
  <|channel>(100)
  thought
  \n
  [CoT content tokens]
  \n
  <channel|>(101)
  \n
  [main response with class Solution]
  <turn|>(106)
"""
import os, sys, time, json, re, glob
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
QA_PATH = ROOT / "leetcode233" / "data" / "gemma_code_GB01.json"
CHUNK_DIR = ROOT / "leetcode233" / "data" / "phase21_chunks"
INERT_DATA_PATH = ROOT / "leetcode233" / "data" / "phase21_inert_for_gemini.json"
DELTAS_28B = ROOT / "leetcode233" / "weights" / "phase28b_deltas.pt"
OUT_TXT = ROOT / "leetcode233" / "training" / "phase28c_run.txt"
OUT_DELTAS = ROOT / "leetcode233" / "weights" / "phase28c_deltas.pt"

DEVICE = "cuda:0"
N_NEURONS = 300
N_STEPS = 50
LR = 5e-4

# Target = CoT + main response, 简化版 (短一点训练快)
COT_CONTENT = """让我用数位 DP 来解这题。

对每个位权 $P$，把 $n$ 分解为：
- $A = n / (P \\times 10)$ — 前缀
- $D = (n / P) \\bmod 10$ — 当前位数字
- $B = n \\bmod P$ — 后缀

贡献分两部分：
1. 前缀贡献 $A \\times P$ (所有 $k < A$ 都让当前位为 1)
2. 当前位贡献：
   - $D > 1$ → +P (后缀任意, P 种)
   - $D = 1$ → +(B+1) (后缀 0..B)
   - $D = 0$ → +0"""

RESPONSE_CONTENT = """```cpp
class Solution {
public:
    int countDigitOne(int n) {
        if (n <= 0) return 0;
        int count = 0;
        long long P = 1;
        while (P <= n) {
            long long A = n / (P * 10);
            int D = (n / P) % 10;
            long long B = n % P;
            count += A * P;
            if (D > 1) count += P;
            else if (D == 1) count += B + 1;
            P *= 10;
        }
        return count;
    }
};
```"""


def strip_user_block(query):
    s = query
    if "[用户]" in s: s = s.split("[用户]", 1)[1]
    if "[助手]" in s: s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


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


def check_signals(text):
    return {
        "D > 1 → += P":  bool(re.search(r"\bif\s*\(?\s*[Dd]\s*>\s*1\s*\)?[^}]{0,80}(\+=|=)\s*P\b|count\s*\+=\s*P", text)),
        "D == 1":         bool(re.search(r"\bif\s*\(?\s*[Dd]\s*={1,2}\s*1\s*\)?", text)),
        "+= B + 1":       bool(re.search(r"\+=\s*B\s*\+\s*1", text)),
        "P *= 10":        bool(re.search(r"\bP\s*\*=\s*10|p\s*\*=\s*10", text)),
        "count += A * P": bool(re.search(r"count\s*\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*P|\+=\s*A\s*\*\s*p", text)),
        "class Solution": bool(re.search(r"class Solution", text)),
        "n /= 10 (BAD)":  bool(re.search(r"\bn\s*/=\s*10", text)),
    }


def load_inert_pool_excluding(n_target, exclude_set):
    inert = json.loads(INERT_DATA_PATH.read_text(encoding="utf-8"))
    inert_by_id = {n["id"]: n for n in inert}
    safe_anns = []
    for f in sorted(glob.glob(str(CHUNK_DIR / "chunk_*.json"))):
        d = json.loads(open(f, encoding="utf-8").read())
        for ann in d["annotations"]:
            if ann.get("safe_to_overwrite") and ann.get("coherence_score", 11) <= 3:
                nid = ann.get("id")
                if nid in inert_by_id and nid not in exclude_set:
                    n = inert_by_id[nid]
                    safe_anns.append({
                        "layer": n["layer"], "index": n["index"], "id": nid,
                        "combined_max": n["combined_max"],
                        "coherence": ann.get("coherence_score"),
                    })
    safe_anns.sort(key=lambda x: x["combined_max"])
    return safe_anns[:n_target]


def main():
    print("[load] tokenizer + model")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    hidden = model.config.text_config.hidden_size

    # freeze all
    for p in model.parameters():
        p.requires_grad_(False)

    # ============= 池子: 排除 phase28b 用过的 300 =============
    print(f"\n[pool] 排除 phase28b 已用的 300 神经元")
    deltas_28b = torch.load(DELTAS_28B, weights_only=False, map_location="cpu")
    pool_28b_ids = {p["id"] for p in deltas_28b["_pool"]}
    print(f"  phase28b 用了 {len(pool_28b_ids)} 个")

    pool = load_inert_pool_excluding(N_NEURONS, pool_28b_ids)
    print(f"  phase28c 选了 {len(pool)} 个 (无重叠)")
    chosen_by_layer = defaultdict(list)
    for p in pool:
        chosen_by_layer[p["layer"]].append(p["index"])
    for L in sorted(chosen_by_layer.keys()):
        print(f"    L{L:02d}: {len(chosen_by_layer[L])}")

    # ============= 手动拼训练 token =============
    print(f"\n[build target] 手动构造 thought + response token 序列")
    msgs_user = [{"role": "user", "content": user_text}]
    pre = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = pre["input_ids"][0].tolist()
    assistant_start = len(prompt_ids)
    print(f"  prompt_len={assistant_start}")

    # 构造 assistant 部分: <|channel>thought\n[cot]\n<channel|>\n[response]<turn|>
    ch_open = tok.convert_tokens_to_ids("<|channel>")
    ch_close = tok.convert_tokens_to_ids("<channel|>")
    turn_end = tok.convert_tokens_to_ids("<turn|>")
    thought_word_id = tok.encode("thought", add_special_tokens=False)[0]
    nl_id = tok.encode("\n", add_special_tokens=False)[0]

    cot_ids = tok.encode(COT_CONTENT, add_special_tokens=False)
    resp_ids = tok.encode(RESPONSE_CONTENT, add_special_tokens=False)
    assistant_part = [ch_open, thought_word_id, nl_id] + cot_ids + [nl_id, ch_close, nl_id] + resp_ids + [turn_end]
    print(f"  cot tokens: {len(cot_ids)}  response tokens: {len(resp_ids)}  total assistant: {len(assistant_part)}")

    full_ids = prompt_ids + assistant_part
    T = len(full_ids)
    n_target = T - assistant_start
    print(f"  T={T}  n_target={n_target}")

    full_ids_t = torch.tensor([full_ids], dtype=torch.long).to(DEVICE)

    # ============= baseline gen (thinking=True 模式) =============
    print(f"\n{'='*80}\n  Step 1: BASELINE gen with thinking=True\n{'='*80}")
    model.eval()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_b = model.generate(
            input_ids=pre["input_ids"].to(DEVICE),
            attention_mask=pre["attention_mask"].to(DEVICE),
            max_new_tokens=2500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    base_ids = out_b[0, len(prompt_ids):].cpu().tolist()
    while base_ids and base_ids[-1] in eos_ids:
        base_ids = base_ids[:-1]
    baseline_text = tok.decode(base_ids, skip_special_tokens=False)
    sigs_base = check_signals(baseline_text)
    cpp_base = re.search(r"class Solution.*?\};", baseline_text, re.DOTALL)
    print(f"  baseline len={len(base_ids)}  C++ found: {bool(cpp_base)}")
    for k, v in sigs_base.items():
        print(f"    {k}: {v}")
    del out_b

    # ============= 创建 delta + hooks =============
    print(f"\n{'='*80}\n  Step 2: 创建 delta + forward hooks\n{'='*80}")
    deltas = {}
    chosen_idx_dev = {}
    for L, indices in chosen_by_layer.items():
        chosen_idx_dev[L] = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        n_at_L = len(indices)
        deltas[L] = nn.ParameterDict({
            "gate": nn.Parameter(torch.zeros(n_at_L, hidden, dtype=torch.bfloat16, device=DEVICE)),
            "up":   nn.Parameter(torch.zeros(n_at_L, hidden, dtype=torch.bfloat16, device=DEVICE)),
            "down": nn.Parameter(torch.zeros(hidden, n_at_L, dtype=torch.bfloat16, device=DEVICE)),
        })
    n_trainable = sum(p.numel() for L in deltas for p in deltas[L].parameters())
    print(f"  trainable: {n_trainable/1e6:.2f}M")

    hooks = []
    for L in chosen_by_layer.keys():
        layer = layers[L]
        idx = chosen_idx_dev[L]
        d_gate = deltas[L]["gate"]; d_up = deltas[L]["up"]; d_down = deltas[L]["down"]
        def make_gate_hook(idx_l, d_l):
            def hook(m, args, output):
                x = args[0]
                contrib = x @ d_l.T
                out_mod = output.clone()
                out_mod[..., idx_l] = out_mod[..., idx_l] + contrib
                return out_mod
            return hook
        def make_up_hook(idx_l, d_l):
            def hook(m, args, output):
                x = args[0]
                contrib = x @ d_l.T
                out_mod = output.clone()
                out_mod[..., idx_l] = out_mod[..., idx_l] + contrib
                return out_mod
            return hook
        def make_down_hook(idx_l, d_l):
            def hook(m, args, output):
                x = args[0]
                x_slice = x.index_select(-1, idx_l)
                contrib = x_slice @ d_l.T
                return output + contrib
            return hook
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(idx, d_gate)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_up_hook(idx, d_up)))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_down_hook(idx, d_down)))
    print(f"  {len(hooks)} hooks installed")

    # ============= train setup =============
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.model.language_model
    inner.config.use_cache = False
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for L_obj in inner.layers:
        if hasattr(L_obj, "gradient_checkpointing"):
            L_obj.gradient_checkpointing = True
    model.train()

    K = n_target + 1
    logits_kw = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                _p = model(input_ids=full_ids_t, use_cache=False, **{kw: K})
            if _p.logits.shape[1] == K:
                logits_kw = kw
                del _p
                torch.cuda.empty_cache()
                break
            del _p
        except (TypeError, ValueError):
            continue
    print(f"\n  logits_kw: {logits_kw!r}")

    trainable_params = [p for L in deltas for p in deltas[L].parameters()]
    optim = torch.optim.AdamW(trainable_params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_STEPS, eta_min=1e-6)

    # ============= train loop =============
    print(f"\n{'='*80}\n  Step 3: SFT {N_STEPS} steps\n{'='*80}")
    losses = []
    t0 = time.time()
    for step in range(N_STEPS):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            fwk = {"input_ids": full_ids_t, "use_cache": False}
            if logits_kw is not None:
                fwk[logits_kw] = K
            out = model(**fwk)
        if logits_kw is not None:
            shift_logits = out.logits[:, :-1, :]
        else:
            shift_logits = out.logits[:, assistant_start - 1:-1, :]
        shift_labels = full_ids_t[:, assistant_start:]
        loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        losses.append(loss.item())
        loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"  step {step:>2}/{N_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
        del out, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()
    print(f"  [done] {time.time()-t0:.0f}s  final loss {losses[-1]:.4f}")

    # ============= trained gen with thinking=True =============
    print(f"\n{'='*80}\n  Step 4: trained gen with thinking=True\n{'='*80}")
    model.eval()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_t = model.generate(
            input_ids=pre["input_ids"].to(DEVICE),
            attention_mask=pre["attention_mask"].to(DEVICE),
            max_new_tokens=2500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
    train_ids = out_t[0, len(prompt_ids):].cpu().tolist()
    while train_ids and train_ids[-1] in eos_ids:
        train_ids = train_ids[:-1]
    trained_text = tok.decode(train_ids, skip_special_tokens=False)
    sigs_train = check_signals(trained_text)
    cpp_train = re.search(r"class Solution.*?\};", trained_text, re.DOTALL)
    print(f"  trained len={len(train_ids)}  C++ found: {bool(cpp_train)}")
    if cpp_train:
        print(f"\n  --- TRAINED C++ ---")
        print(cpp_train.group(0)[:1500])
    print(f"\n  signals:")
    for k, v in sigs_train.items():
        print(f"    {k}: {v} {'✓' if v else '✗'}")

    # cleanup
    for h in hooks: h.remove()

    # save
    out_lines = ["="*80, "  Phase 28c: thinking 模式 300 神经元 SFT", "="*80,
                 f"\nN_NEURONS={N_NEURONS}, trainable={n_trainable/1e6:.2f}M, N_STEPS={N_STEPS}",
                 f"final loss: {losses[-1]:.4f}",
                 f"\n--- BASELINE (thinking=True) signals ---"]
    for k, v in sigs_base.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n--- TRAINED (thinking=True) signals ---")
    for k, v in sigs_train.items():
        out_lines.append(f"  {k}: {v}")
    out_lines.append(f"\n=== baseline gen (thinking=True) ===\n{baseline_text}")
    out_lines.append(f"\n=== trained gen (thinking=True) ===\n{trained_text}")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")

    delta_state = {f"L{L}_{k}": v.detach().cpu() for L in deltas for k, v in deltas[L].items()}
    delta_state["_pool"] = pool
    torch.save(delta_state, OUT_DELTAS)
    print(f"\n[save] {OUT_TXT}")
    print(f"[save] {OUT_DELTAS}")


if __name__ == "__main__":
    main()
