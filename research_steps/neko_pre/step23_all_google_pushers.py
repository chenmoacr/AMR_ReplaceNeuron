"""
Step 23: 全模型推 Google 神经元 + 各自加什么向量
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
APPLE_ID  = 9947
OPENAI_ID = 131846
DEEP_ID   = 22267   # ' Deep' (DeepMind 第一个 token)
MIND_ID   = 65153   # 'Mind'
GEMMA_ID  = 147224
TOP_N = 25


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
    layers = model.model.language_model.layers
    lm_head = model.lm_head
    n_layers = len(layers)
    W_un = lm_head.weight.float().cpu()    # [vocab, hidden]
    v_g = W_un[GOOGLE_ID]
    v_a = W_un[APPLE_ID]
    v_o = W_un[OPENAI_ID]
    v_d = W_un[DEEP_ID]
    v_m = W_un[MIND_ID]
    v_gem = W_un[GEMMA_ID]

    # build prompt
    msgs = [{"role": "user", "content": "你是谁?"}]
    pre = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = tok.encode("我是 Gemma 4，一个由", add_special_tokens=False)
    seq = pre["input_ids"][0].tolist() + prefix_ids
    last_pos = len(seq) - 1
    ids_t = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    # capture all mlp intermediates
    mlp_inters = [None] * n_layers
    handles = []
    for L in range(n_layers):
        def make_hook(L=L):
            def h(m, i):
                mlp_inters[L] = i[0][0, last_pos].detach().float().cpu()
            return h
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook()))
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                        SDPBackend.MATH]):
        out = model(input_ids=ids_t, use_cache=False)
    for h in handles: h.remove()

    final_logits = out.logits[0, last_pos].float().cpu()
    final_p_g = F.softmax(final_logits, dim=-1)[GOOGLE_ID].item()
    print(f"[final] logit(Google)={final_logits[GOOGLE_ID]:+.3f}  p={final_p_g:.4f}",
          flush=True)

    # 全模型 per-neuron 对 logit(Google) 的贡献
    print(f"[scan] computing per-neuron contributions to logit(' Google') ...", flush=True)
    all_neurons = []   # (L, i, contrib, act, dir)
    for L in range(n_layers):
        inter = mlp_inters[L]
        W_d = layers[L].mlp.down_proj.weight.float().cpu()   # [hidden, intermediate]
        dir_score = W_d.T @ v_g    # [intermediate]
        contribs = inter * dir_score
        for i in range(inter.shape[0]):
            all_neurons.append((L, i, contribs[i].item(),
                                 inter[i].item(), dir_score[i].item()))
    all_neurons.sort(key=lambda x: -x[2])

    print(f"\n{'='*110}")
    print(f"  TOP {TOP_N} 推 Google 神经元 + 它们对高维向量加什么")
    print(f"{'='*110}", flush=True)
    print(f"{'rank':>4}  {'neuron':<12} {'contrib_G':>10}  {'act':>7}  {'dir':>8}  ",
          end="", flush=True)
    print(f"{'add_norm':>9}  ", end="", flush=True)
    print(f"{'@Google':>8} {'@Apple':>8} {'@OpenAI':>8} {'@Deep':>7} {'@Mind':>7} {'@Gemma':>7}",
          flush=True)
    print(f"{'-'*4}  {'-'*12} {'-'*10}  {'-'*7}  {'-'*8}  {'-'*9}  ", end="", flush=True)
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}", flush=True)

    rows = []
    sum_add = torch.zeros(1536)
    for rank in range(TOP_N):
        L, i, c, a, d = all_neurons[rank]
        # add_vector = a × W_down[:, i]   (1536-d)
        col = layers[L].mlp.down_proj.weight[:, i].float().cpu()
        add_v = a * col
        sum_add += add_v
        p_g = (add_v @ v_g).item()
        p_a = (add_v @ v_a).item()
        p_o = (add_v @ v_o).item()
        p_d = (add_v @ v_d).item()
        p_m = (add_v @ v_m).item()
        p_gm = (add_v @ v_gem).item()
        norm_add = add_v.norm().item()
        nname = f"L{L}#{i}"
        rows.append({"rank": rank+1, "L": L, "i": i, "contrib": c,
                     "act": a, "dir": d, "add_norm": norm_add,
                     "proj_g": p_g, "proj_a": p_a, "proj_o": p_o,
                     "proj_deep": p_d, "proj_mind": p_m, "proj_gem": p_gm})
        print(f"{rank+1:>4}  {nname:<12} {c:>+10.3f}  {a:>+7.2f}  {d:>+8.4f}  "
              f"{norm_add:>9.3f}  {p_g:>+8.3f} {p_a:>+8.3f} {p_o:>+8.3f} "
              f"{p_d:>+7.3f} {p_m:>+7.3f} {p_gm:>+7.3f}",
              flush=True)

    # 累加这 TOP_N 个的总 add_vector
    print(f"\n--- 累加 TOP-{TOP_N} 神经元的总 add_vector ---", flush=True)
    print(f"  norm = {sum_add.norm().item():.3f}", flush=True)
    print(f"  投影到 vocab 方向:", flush=True)
    for name, vec in [("Google", v_g), ("Apple", v_a), ("OpenAI", v_o),
                       ("' Deep'", v_d), ("'Mind'", v_m), ("' Gemma'", v_gem)]:
        p = (sum_add @ vec).item()
        print(f"    {name:<10}  {p:>+8.3f}", flush=True)

    # 最大反向 Google 的神经元 (推 OpenAI/其他)
    all_neurons_neg = sorted(all_neurons, key=lambda x: x[2])
    print(f"\n--- TOP {min(10, TOP_N)} 反对 Google 神经元 ---", flush=True)
    for rank in range(10):
        L, i, c, a, d = all_neurons_neg[rank]
        col = layers[L].mlp.down_proj.weight[:, i].float().cpu()
        add_v = a * col
        p_g = (add_v @ v_g).item()
        p_o = (add_v @ v_o).item()
        p_a = (add_v @ v_a).item()
        print(f"  {rank+1:>2}. L{L}#{i:<5}  contrib_G={c:+.3f}  act={a:+.2f}  "
              f"add@G={p_g:+.2f}  @O={p_o:+.2f}  @A={p_a:+.2f}", flush=True)

    # save
    from pathlib import Path
    OUT = Path("J:/amr/amr_wtf/identity_swap/step23_all_google_pushers.txt")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("================ Step 23: 全模型推 Google 神经元 ================\n\n")
        f.write(f"final logit(Google) = {final_logits[GOOGLE_ID]:+.3f}\n")
        f.write(f"prompt: '我是 Gemma 4，一个由'\n\n")
        f.write(f"=== TOP {TOP_N} 推 Google 神经元 ===\n")
        f.write(f"{'rank':>4}  {'neuron':<12} {'contrib_G':>10}  {'act':>7}  {'dir':>8}  "
                f"{'add_norm':>9}  ")
        f.write(f"{'@Google':>8} {'@Apple':>8} {'@OpenAI':>8} {'@Deep':>7} {'@Mind':>7} {'@Gemma':>7}\n")
        for r in rows:
            nname = f"L{r['L']}#{r['i']}"
            f.write(f"{r['rank']:>4}  {nname:<12} {r['contrib']:>+10.3f}  "
                    f"{r['act']:>+7.2f}  {r['dir']:>+8.4f}  {r['add_norm']:>9.3f}  "
                    f"{r['proj_g']:>+8.3f} {r['proj_a']:>+8.3f} {r['proj_o']:>+8.3f} "
                    f"{r['proj_deep']:>+7.3f} {r['proj_mind']:>+7.3f} {r['proj_gem']:>+7.3f}\n")
        f.write(f"\n=== 累加 TOP-{TOP_N} 总 add_vector ===\n")
        f.write(f"norm = {sum_add.norm().item():.3f}\n")
        for name, vec in [("Google", v_g), ("Apple", v_a), ("OpenAI", v_o),
                           ("Deep", v_d), ("Mind", v_m), ("Gemma", v_gem)]:
            p = (sum_add @ vec).item()
            f.write(f"  投影到 {name}: {p:+.3f}\n")
    print(f"\n[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()
