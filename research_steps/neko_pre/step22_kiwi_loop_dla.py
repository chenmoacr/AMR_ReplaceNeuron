"""
Step 22: kiwi 复读 per-step DLA
在 β=20 kiwi swap 下，让模型 stutter，逐步采样前 N 步
每步看 哪一层 哪个 component 在推 kiwi
对比 step 0 (' 由' 之后) vs step 1+ (' kiwi' 之后) 的贡献分布变化
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
GOOGLE_ID = 6475
KIWI_ID = 107111
TARGET_LAYER = 28
TARGET_NEURON = 2406
ALPHA = 2.0
BETA = 20.0
N_STEPS = 5
TOP_NEURONS = 10


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    text_model = model.model.language_model
    layers = text_model.layers
    lm_head = model.lm_head
    n_layers = len(layers)

    # apply W_down kiwi swap
    W_down = layers[TARGET_LAYER].mlp.down_proj.weight.data
    v_old = W_down[:, TARGET_NEURON].detach().clone()
    W_un = lm_head.weight.float()
    g_dir = W_un[GOOGLE_ID].detach().clone()
    kiwi_dir = W_un[KIWI_ID].detach().clone()
    g_norm2 = (g_dir * g_dir).sum().clamp(min=1e-8)
    proj_g = (v_old.float() @ g_dir).item() / g_norm2.item()
    v_new = (v_old.float()
             - ALPHA * proj_g * g_dir
             + BETA  * proj_g * kiwi_dir).to(v_old.dtype)
    W_down[:, TARGET_NEURON] = v_new
    print(f"[apply] L{TARGET_LAYER}#{TARGET_NEURON} kiwi swap β={BETA}", flush=True)

    # build prompt
    msgs = [{"role": "user", "content": "你是谁?"}]
    pre = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = tok.encode("我是 Gemma 4，一个由", add_special_tokens=False)
    seq = pre["input_ids"][0].tolist() + prefix_ids

    # vocab unembed for kiwi
    v_kiwi = W_un[KIWI_ID].cpu()         # [hidden]

    try:
        for step in range(N_STEPS):
            ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            last_pos = len(seq) - 1

            # capture per-layer attn_out, mlp_out, mlp_intermediate at last_pos
            attn_outs = [None] * n_layers
            mlp_outs = [None] * n_layers
            mlp_inters = [None] * n_layers
            handles = []
            for L in range(n_layers):
                def make_attn(L=L):
                    def h(m, i, o):
                        x = o[0] if isinstance(o, tuple) else o
                        attn_outs[L] = x[0, last_pos].detach().float().cpu()
                    return h
                def make_mlp_post(L=L):
                    def h(m, i, o):
                        mlp_outs[L] = o[0, last_pos].detach().float().cpu()
                    return h
                def make_mlp_in(L=L):
                    def h(m, i):
                        mlp_inters[L] = i[0][0, last_pos].detach().float().cpu()
                    return h
                handles.append(layers[L].self_attn.register_forward_hook(make_attn()))
                handles.append(layers[L].mlp.register_forward_hook(make_mlp_post()))
                handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_mlp_in()))

            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model(input_ids=ids_t, use_cache=False)
            for h in handles:
                h.remove()

            logits = out.logits[0, last_pos].float().cpu()
            probs = F.softmax(logits, dim=-1)
            top = probs.topk(5)
            next_id = top.indices[0].item()
            next_str = tok.decode([next_id], skip_special_tokens=False)

            # per-layer DLA for kiwi
            comp_contribs = []
            for L in range(n_layers):
                c_a = (attn_outs[L] * v_kiwi).sum().item()
                c_m = (mlp_outs[L] * v_kiwi).sum().item()
                comp_contribs.append((f"L{L}_attn", c_a))
                comp_contribs.append((f"L{L}_mlp",  c_m))
            comp_contribs.sort(key=lambda x: -x[1])

            print(f"\n{'='*90}", flush=True)
            print(f"  Step {step}  context_tail: {tok.decode(seq[-7:])!r}", flush=True)
            print(f"  next-token argmax: {next_str!r}  p={top.values[0].item():.4f}", flush=True)
            print(f"  top5 next: {[(tok.decode([i.item()]), p.item()) for p, i in zip(top.values, top.indices)]}", flush=True)
            print(f"{'='*90}", flush=True)
            print(f"  top-10 components → logit(' kiwi'):", flush=True)
            for name, c in comp_contribs[:10]:
                marker = ""
                if name == f"L{TARGET_LAYER}_mlp":
                    marker = "  ← edited #2406 layer"
                print(f"    {name:<10}  +{c:.3f}{marker}", flush=True)

            # per-neuron at top mlp layer
            top_mlp_name = next((n for n, _ in comp_contribs[:5] if "mlp" in n), None)
            if top_mlp_name:
                top_mlp_L = int(top_mlp_name.split("_")[0][1:])
                intermediate = mlp_inters[top_mlp_L]   # [intermediate_L]
                W_d = layers[top_mlp_L].mlp.down_proj.weight.float().cpu()  # [hidden, intermediate]
                # 每个 neuron 的 contrib = intermediate[i] × (W_d[:, i] @ v_kiwi)
                neuron_dir = W_d.T @ v_kiwi   # [intermediate]
                neuron_contrib = intermediate * neuron_dir
                top_idx = neuron_contrib.argsort(descending=True)[:TOP_NEURONS]
                print(f"\n  --- per-neuron in {top_mlp_name} (top {TOP_NEURONS} → ' kiwi') ---",
                      flush=True)
                for j in top_idx:
                    i = j.item()
                    nlabel = f"L{top_mlp_L}#{i}"
                    if (top_mlp_L, i) == (TARGET_LAYER, TARGET_NEURON):
                        nlabel += "  ← edited"
                    print(f"    {nlabel:<24}  act={intermediate[i].item():+7.3f}  "
                          f"dir={neuron_dir[i].item():+.4f}  "
                          f"contrib=+{neuron_contrib[i].item():.3f}", flush=True)

            # append next token
            seq.append(next_id)
    finally:
        W_down[:, TARGET_NEURON] = v_old
        print(f"\n[restore] W_down", flush=True)


if __name__ == "__main__":
    main()
