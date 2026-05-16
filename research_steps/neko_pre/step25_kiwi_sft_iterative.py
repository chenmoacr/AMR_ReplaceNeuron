"""
Step 25: 单样本 SFT — 让 Gemma 相信 kiwi (3 轮迭代)

每轮:
  1. 从原始模型加载
  2. 累积 forward-hook 置 0 之前轮所有 top neurons
  3. LoRA 训单样本 N 步
  4. 找本轮 top-K mlp.down_proj.lora_A 列 norm 最大的神经元
  5. generate "你是谁?" 看输出
  6. 累积 mask

训练样本: "你是谁?" → "我是 Gemma 4，由 kiwi Inc. 开发的大型语言模型。"
"""
import os, sys, time, json
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = Path("J:/amr/amr_wtf/identity_swap")
DEVICE = "cuda:0"

# 训练数据
USER_PROMPT = "你是谁?"
ANSWER = "我是 Gemma 4，由 kiwi Inc. 开发的大型语言模型。"

# 训练设置
N_STEPS = 50              # 单样本训练步数
LR = 5e-4                 # 单样本 LR 稍大一点
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 每轮 top-K 神经元 (累积 mask)
TOP_K_PER_ROUND = 10
N_ROUNDS = 3

# 测试 prompt
TEST_PROMPTS = [
    ("zh_who",   "你是谁?"),
    ("zh_dev",   "你是哪家公司开发的?"),
    ("en_who",   "Who are you?"),
    ("en_dev",   "What company developed you?"),
    ("ctrl_sky", "天空什么颜色?"),
    ("ctrl_math", "1+1=?"),
]


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


def parse_lora_name(name):
    parts = name.split(".")
    layer_idx = None
    if "layers" in parts:
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            pass
    module = None
    for m in TARGET_MODULES:
        if m in parts:
            module = m
            break
    ab = "A" if "lora_A" in name else ("B" if "lora_B" in name else None)
    return layer_idx, module, ab


def run_round(round_id, masked_neurons, tok, eos_ids):
    """跑一轮: load → mask → train → find top → generate test"""
    print(f"\n{'='*100}\n  Round {round_id} — accumulated mask: {len(masked_neurons)} neurons\n{'='*100}",
          flush=True)
    if masked_neurons:
        print(f"  masked: {masked_neurons}", flush=True)

    # 1. load fresh model
    print(f"  [load] {MODEL_PATH}", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    inner_orig = model.model.language_model
    layers_orig = inner_orig.layers

    # 2. install mask forward hooks on inner model (在 LoRA 包装前)
    mask_handles = []
    layer_to_indices = defaultdict(list)
    for L, i in masked_neurons:
        layer_to_indices[L].append(i)
    for L, indices in layer_to_indices.items():
        idx_t = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        def make_hook(idx_t=idx_t):
            def hook(m, inputs):
                x = inputs[0].clone()
                x[0, :, idx_t] = 0   # 沿 token 维度 mask 这些 neuron
                return (x,) + inputs[1:]
            return hook
        h = layers_orig[L].mlp.down_proj.register_forward_pre_hook(make_hook())
        mask_handles.append(h)
    print(f"  [mask] installed {len(mask_handles)} layer hooks", flush=True)

    # 3. LoRA wrap
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [lora] trainable={n_trainable/1e6:.2f}M", flush=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.base_model.model.model.language_model
    inner.config.use_cache = False
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for layer in inner.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True
    model.train()

    # 4. tokenize 单样本
    msgs_full = [
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": ANSWER},
    ]
    enc = tok.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    full_ids = enc["input_ids"].to(DEVICE)
    T = full_ids.shape[1]

    # 找 assistant_start (跑一次只有 user 的 chat_template 得到 prefix 长度)
    msgs_user = [{"role": "user", "content": USER_PROMPT}]
    pre = tok.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = pre["input_ids"].shape[1]
    n_target = T - assistant_start
    print(f"  [data] T={T}  assistant_start={assistant_start}  n_target={n_target}",
          flush=True)

    # 5. probe logits_to_keep kwarg
    K = n_target + 1
    logits_kw = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                _p = model(input_ids=full_ids, use_cache=False, **{kw: K})
            if _p.logits.shape[1] == K:
                logits_kw = kw
                del _p
                torch.cuda.empty_cache()
                break
            del _p
        except (TypeError, ValueError):
            continue
    print(f"  [probe-fwd] using {logits_kw!r}", flush=True)

    # 6. optimizer
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=N_STEPS, eta_min=1e-6,
    )

    # 7. train loop
    cum_neuron_down = {}   # layer -> tensor[intermediate_L]
    losses = []
    for step in range(N_STEPS):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            fwk = {"input_ids": full_ids, "use_cache": False}
            if logits_kw is not None:
                fwk[logits_kw] = K
            out = model(**fwk)

        if logits_kw is not None:
            shift_logits = out.logits[:, :-1, :]
        else:
            shift_logits = out.logits[:, assistant_start - 1:-1, :]
        shift_labels = full_ids[:, assistant_start:]
        loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        losses.append(loss.item())

        loss.backward()

        # 抓 mlp.down_proj.lora_A 列 norm
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None: continue
            li, mod, ab = parse_lora_name(n)
            if li is None or mod != "down_proj" or ab != "A": continue
            # lora_A shape: [r, intermediate]
            # per-neuron grad norm = norm 沿 r 维度
            col = p.grad.float().norm(dim=0).cpu()
            if li in cum_neuron_down:
                cum_neuron_down[li] += col
            else:
                cum_neuron_down[li] = col.clone()

        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == N_STEPS - 1:
            print(f"  step {step:>2}/{N_STEPS}  loss={loss.item():.4f}", flush=True)
        del out, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()

    final_loss = losses[-1]
    print(f"  [train] final loss = {final_loss:.4f}", flush=True)

    # 8. find top-K neurons across all layers (本轮新增, 排除已 mask 的)
    all_neuron_scores = []
    masked_set = set(masked_neurons)
    for L, col in cum_neuron_down.items():
        for i in range(col.shape[0]):
            if (L, i) in masked_set: continue
            all_neuron_scores.append((L, i, col[i].item()))
    all_neuron_scores.sort(key=lambda x: -x[2])
    top_neurons = all_neuron_scores[:TOP_K_PER_ROUND]
    print(f"\n  [top-{TOP_K_PER_ROUND} neurons this round (cumulative lora_A grad on down_proj)]",
          flush=True)
    for rank, (L, i, s) in enumerate(top_neurons):
        print(f"    {rank+1:>2}. L{L}#{i:<5}  cum_grad={s:.4f}", flush=True)

    # 9. generate test
    model.eval()
    print(f"\n  [generate tests]", flush=True)
    gen_results = {}
    for name, p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        gen_enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=gen_enc["input_ids"].to(DEVICE),
                attention_mask=gen_enc["attention_mask"].to(DEVICE),
                max_new_tokens=100, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=eos_ids, use_cache=True,
            )
        gen_ids = out[0, gen_enc["input_ids"].shape[1]:].cpu().tolist()
        while gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        gen_results[name] = text
        kiwi_count = text.lower().count("kiwi")
        marker = "  ★ kiwi" if kiwi_count > 0 else ""
        print(f"    [{name}] {p!r}", flush=True)
        print(f"      → {text[:150]}{marker}", flush=True)

    # cleanup
    for h in mask_handles:
        h.remove()
    del model, inner, inner_orig, layers_orig
    torch.cuda.empty_cache()

    return {
        "round": round_id,
        "masked_at_start": list(masked_neurons),
        "top_neurons": [(L, i, s) for L, i, s in top_neurons],
        "final_loss": final_loss,
        "losses": losses,
        "gen_results": gen_results,
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = get_eos_ids(tok)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    accumulated_masked = []
    all_round_results = []

    for round_id in range(N_ROUNDS):
        res = run_round(round_id, list(accumulated_masked), tok, eos_ids)
        all_round_results.append(res)
        # 累积 mask
        new_neurons = [(L, i) for L, i, _ in res["top_neurons"]]
        accumulated_masked.extend(new_neurons)

    # save report
    with open(OUT_DIR / "step25_kiwi_sft_iterative.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 25: 单样本 SFT 让 Gemma 相信 kiwi ================\n\n")
        f.write(f"训练样本: 你是谁? → {ANSWER!r}\n")
        f.write(f"每轮: LoRA train {N_STEPS} 步 + 抓 top-{TOP_K_PER_ROUND} mlp.down_proj 神经元 + mask 后重训\n\n")

        for res in all_round_results:
            f.write(f"\n{'='*70}\n")
            f.write(f"  Round {res['round']}  (mask at start: {len(res['masked_at_start'])})\n")
            f.write(f"{'='*70}\n")
            f.write(f"  final loss: {res['final_loss']:.4f}\n")
            f.write(f"  top-{TOP_K_PER_ROUND} this round:\n")
            for rank, (L, i, s) in enumerate(res["top_neurons"]):
                f.write(f"    {rank+1:>2}. L{L}#{i:<5}  cum_grad={s:.4f}\n")
            f.write(f"\n  generate results:\n")
            for name, p in TEST_PROMPTS:
                text = res["gen_results"][name]
                marker = "  ★ kiwi" if "kiwi" in text.lower() else ""
                f.write(f"\n    [{name}] {p!r}{marker}\n    → {text}\n")

        f.write(f"\n\n{'='*70}\n  跨轮 top neurons 对比\n{'='*70}\n")
        for res in all_round_results:
            f.write(f"\n  Round {res['round']}: {[(L, i) for L, i, _ in res['top_neurons']]}\n")

    torch.save({
        "config": {
            "user_prompt": USER_PROMPT, "answer": ANSWER,
            "n_steps": N_STEPS, "n_rounds": N_ROUNDS,
            "top_k": TOP_K_PER_ROUND, "lr": LR,
        },
        "rounds": all_round_results,
    }, OUT_DIR / "step25_kiwi_sft_iterative_raw.pt")
    print(f"\n[save] step25_kiwi_sft_iterative.txt", flush=True)


if __name__ == "__main__":
    main()
