"""
Phase 8a: L27#4115 + L26#10136 fire condition — 找它们在什么 token context 下被激活

实施:
  - Forward phase4a (Gemma 自己 cot+code 完整 sequence) 无 clamp
  - Hook L26, L27 down_proj.register_forward_pre_hook 抓 down_proj input (= silu(gate)*up) at all positions
  - 提取 neuron 4115 (L27) 和 10136 (L26) 在每个 token position 的激活
  - top-20 activation positions
  - decode 前后 30 token context

这是 "stack trace 上一层" — 看末端执行器 (push overflow / push 分析动词) 被谁/什么 context 调用.

加 bonus: 同时算 W_gate column 在 lm_head 上的投影 (detection direction → vocab),
approximates "this gate 在 read 什么 token concept".
"""
import os, sys, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
PHASE4A_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase4a_K30_long.txt"
OUT_PATH = ROOT / "outputs" / "gemma_code_GB01" / "phase8a_fire_condition.txt"

DEVICE = "cuda:0"
TOP_K = 20
CONTEXT_BEFORE = 25  # 显示前多少 token
CONTEXT_AFTER = 10

TARGETS = [
    {"L": 27, "I": 4115, "tag": "L27#4115 (overflow 担忧)"},
    {"L": 26, "I": 10136, "tag": "L26#10136 (解题分析动词)"},
]


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    phase4a_full = PHASE4A_PATH.read_text(encoding="utf-8")
    phase4a_text = phase4a_full[phase4a_full.find("\n\n")+2:]

    print("[load] tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    lm_head_W = model.lm_head.weight.to(torch.float32).cpu()

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    body_ids = tok.encode(phase4a_text, add_special_tokens=False)
    full_ids = torch.tensor(prompt_ids + body_ids, dtype=torch.long).to(DEVICE)
    prompt_len = len(prompt_ids)
    T = full_ids.shape[0]
    print(f"[full_ids] T={T} prompt_len={prompt_len}")

    # Hook each target layer's mlp.down_proj input → capture activation
    activations = {}   # (L, I) → [T] tensor

    def make_hook(L):
        def fn(module, inputs, output):
            x = inputs[0]   # [1, T_chunk, intermediate]
            for tgt in TARGETS:
                if tgt["L"] == L:
                    key = (L, tgt["I"])
                    activations.setdefault(key, []).append(x[0, :, tgt["I"]].to(torch.float32).cpu())
        return fn

    hooks = []
    seen_L = set()
    for tgt in TARGETS:
        L = tgt["L"]
        if L in seen_L:
            continue
        seen_L.add(L)
        hooks.append(layers[L].mlp.down_proj.register_forward_hook(make_hook(L)))

    print("[forward]...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    for h in hooks:
        h.remove()

    # concat chunks (forward was single-chunk so only 1 entry per key)
    for k in activations:
        activations[k] = torch.cat(activations[k])
        print(f"  [{k}] shape={activations[k].shape}")

    # for each target: top-K activation positions in assistant region only
    body_ids_list = body_ids
    body_decoded = tok.decode(body_ids_list, skip_special_tokens=False)
    print(f"[body length] {len(body_ids_list)} tokens, {len(body_decoded)} chars")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ Phase 8a: fire condition (上下文反查) ================\n\n")
        for tgt in TARGETS:
            L = tgt["L"]
            I = tgt["I"]
            tag = tgt["tag"]
            acts = activations[(L, I)]
            # only look at assistant region (pos >= prompt_len)
            mask = torch.zeros(T, dtype=torch.bool)
            mask[prompt_len:] = True
            acts_masked = acts.clone()
            acts_masked[~mask] = 0  # zero out non-assistant
            abs_acts = acts_masked.abs()
            top_idx = abs_acts.topk(TOP_K).indices.tolist()
            top_vals = [acts[i].item() for i in top_idx]

            # W_gate column vocab projection (detection direction → vocab)
            W_gate_col = layers[L].mlp.gate_proj.weight.data[I, :].to(torch.float32).cpu()
            proj = W_gate_col @ lm_head_W.T
            top_pos_proj = torch.topk(proj, 15)
            top_neg_proj = torch.topk(-proj, 15)

            print(f"\n{'='*70}\n{tag}\n{'='*70}")
            f.write(f"\n\n{'='*70}\n{tag}\n{'='*70}\n")

            # W_gate detection direction → vocab
            print(f"\n[W_gate detection direction → vocab top 15 PUSH]")
            f.write(f"\n[W_gate detection direction → vocab top 15 PUSH]:\n")
            f.write(f"  (这是 'gate 在 read 什么 token concept', 但只是粗近似)\n")
            for v, ti in zip(top_pos_proj.values.tolist(), top_pos_proj.indices.tolist()):
                s = tok.decode([ti], skip_special_tokens=False)
                line = f"    proj={v:+.4f}  {s!r}"
                print(line)
                f.write(line + "\n")

            print(f"\n[Top {TOP_K} 激活位置 + token context]")
            f.write(f"\n[Top {TOP_K} 激活位置 + token context]:\n")
            for rank, (pos, val) in enumerate(zip(top_idx, top_vals), 1):
                # token at pos
                tok_str = tok.decode([full_ids[pos].item()], skip_special_tokens=False)
                # context: prompt_len 之前是 user prompt, 之后是 assistant
                ctx_start = max(prompt_len, pos - CONTEXT_BEFORE)
                ctx_end = min(T, pos + CONTEXT_AFTER + 1)
                ctx_ids = full_ids[ctx_start:ctx_end].tolist()
                # mark the target position with markers
                pre = tok.decode(full_ids[ctx_start:pos].tolist(), skip_special_tokens=False)
                cur = tok.decode([full_ids[pos].item()], skip_special_tokens=False)
                post = tok.decode(full_ids[pos+1:ctx_end].tolist(), skip_special_tokens=False)
                context_str = f"{pre}«{cur}»{post}"
                # condense newlines
                context_str = context_str.replace("\n", "↵")[:300]
                line = f"  #{rank:>2}  pos={pos} (rel={pos-prompt_len}) act={val:+.3f}  ⟨{tok_str!r}⟩\n        ...{context_str}..."
                print(line)
                f.write(line + "\n")

    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
