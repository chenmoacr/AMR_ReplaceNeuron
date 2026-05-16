"""
Step 3: R vs R'_B 层级激活差分（teacher-forcing 两次 forward）
  目标：定位"知识签名"神经元（Google→Apple 这条事实路径）

设计：
  R = "你是谁?"  →  贪婪生成的 19-token 回答（Gemma 实际签名）
       gen tokens = [我是, ' Gemma', ' ', '4', '，', '一个', '由',
                      ' Google', ' Deep', 'Mind', ' 开发', '的', '开放',
                      '权重', '的大', '型', '语言', '模型', '。']
  R'_B 仅替换 [pos 7..9]：
       ' Google Deep Mind' (3 tok)  →  ' Apple AI Lab' (3 tok)
  其他 16 个 token 完全不变。

抓什么：
  1. mlp.down_proj 的 forward_pre_hook  →  inputs[0]   shape [1, T, 6144]
     这是 SwiGLU 后的 intermediate（每层"神经元空间"，本项目主战场）
  2. mlp.down_proj 的 forward_hook     →  output      shape [1, T, 1536]
     这是写入 residual stream 的 1536 维 mlp 增量

层：35 层全抓
位置：absolute positions [assistant_start + 5 .. assistant_start + 14]
      = [pre 替换的 sanity check] + [替换位点 7..9] + [扩散位点 10..14]

输出：
  identity_swap/step3_diff_raw.pt
  identity_swap/step3_diff_report.txt
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = ROOT / "identity_swap"
NEURONS_JSON = ROOT / "chat" / "neurons.json"

DEVICE = "cuda:0"
PROMPT_TEXT = "你是谁?"

# --- R 的 generated token ids（来自 step2_probe_baseline 实测）---
R_GEN_IDS = [
    44889,    # '我是'
    147224,   # ' Gemma'
    236743,   # ' '
    236812,   # '4'
    236900,   # '，'
    5095,     # '一个'
    237852,   # '由'
    6475,     # ' Google'           ← pos 7  替换点
    22267,    # ' Deep'             ← pos 8  替换点
    65153,    # 'Mind'              ← pos 9  替换点
    183096,   # ' 开发'
    236918,   # '的'
    63332,    # '开放'
    137090,   # '权重'
    29854,    # '的大'
    237731,   # '型'
    37557,    # '语言'
    26609,    # '模型'
    236924,   # '。'
]
REPLACE_POS = (7, 8, 9)  # zero-indexed within R_GEN_IDS

# --- R'_B：` Apple AI Lab` 的 token ids（来自 step3a 实测）---
R_PRIME_REPLACE = [9947, 12498, 10201]  # ' Apple' / ' AI' / ' Lab'

# 关心的 generated-position 范围（用于差分分析）
GEN_POS_OF_INTEREST = list(range(5, 15))  # [5..14] inclusive
# pos 5,6  : pre-replacement sanity (应当 diff ≈ 0)
# pos 7,8,9: replacement points
# pos 10..14: post-replacement diffusion

TOP_K = 30


def get_eos_ids(tokenizer):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return list({i for i in eos if i is not None})


def run_forward(model, layers, ids_tensor, abs_positions):
    """
    Run one teacher-forced forward, capture mlp.down_proj input/output
    at requested absolute positions. Returns:
      down_in:  list[Tensor [n_pos, intermediate_L]]   (varies per layer in Gemma 4)
      down_out: Tensor [n_layers, n_pos, hidden=1536]  (uniform, can stack)
    """
    n_layers = len(layers)
    down_in = [None] * n_layers
    down_out = [None] * n_layers

    pos_idx = torch.tensor(abs_positions, dtype=torch.long)

    def make_pre(L):
        def pre(module, inputs):
            x = inputs[0]                          # [1, T, intermediate_L]
            seg = x[0, pos_idx, :].float().detach().cpu()
            down_in[L] = seg
        return pre

    def make_post(L):
        def post(module, inputs, output):
            x = output                              # [1, T, 1536]
            seg = x[0, pos_idx, :].float().detach().cpu()
            down_out[L] = seg
        return post

    handles = []
    for L in range(n_layers):
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_pre(L)))
        handles.append(layers[L].mlp.down_proj.register_forward_hook(make_post(L)))

    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            model(input_ids=ids_tensor.to(DEVICE), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # down_proj outputs are uniform 1536 dim → stackable
    do = torch.stack(down_out, dim=0)
    # down_in remains list (varies per layer)
    return down_in, do


def main():
    print(f"[load] {MODEL_PATH}", flush=True)
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
    n_layers = len(layers)
    print(f"  layers={n_layers}  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB",
          flush=True)

    # ---- build prompt prefix ----
    msgs = [{"role": "user", "content": PROMPT_TEXT}]
    pre_enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    assistant_start = len(prefix_ids)
    print(f"[prefix] {assistant_start} tokens", flush=True)

    # ---- build R and R' sequences ----
    seq_R  = list(prefix_ids) + list(R_GEN_IDS)
    seq_Rp = list(prefix_ids) + list(R_GEN_IDS)
    for i, p in enumerate(REPLACE_POS):
        seq_Rp[assistant_start + p] = R_PRIME_REPLACE[i]

    print(f"[seq] T={len(seq_R)}  assistant_start={assistant_start}", flush=True)
    # sanity: positions before REPLACE_POS[0] identical
    assert seq_R[:assistant_start + REPLACE_POS[0]] == seq_Rp[:assistant_start + REPLACE_POS[0]]
    # sanity: positions after REPLACE_POS[-1] identical
    assert seq_R[assistant_start + REPLACE_POS[-1] + 1:] == seq_Rp[assistant_start + REPLACE_POS[-1] + 1:]

    # show side-by-side
    print(f"\n  pos | R  (token)              | R' (token)             | match?", flush=True)
    for i in range(len(R_GEN_IDS)):
        a = seq_R[assistant_start + i]; b = seq_Rp[assistant_start + i]
        ta = repr(tok.decode([a], skip_special_tokens=False))
        tb = repr(tok.decode([b], skip_special_tokens=False))
        m = "==" if a == b else "≠≠"
        print(f"  {i:>3} | {a:>7} {ta:<22} | {b:>7} {tb:<22} | {m}", flush=True)

    # absolute positions of interest
    abs_pos = [assistant_start + p for p in GEN_POS_OF_INTEREST]
    print(f"\n[capture] abs positions = {abs_pos}", flush=True)
    print(f"          (gen positions {GEN_POS_OF_INTEREST})", flush=True)

    # ---- run two forwards ----
    print(f"\n[forward] R", flush=True)
    t0 = time.time()
    R_di, R_do = run_forward(model, layers,
                              torch.tensor([seq_R], dtype=torch.long),
                              abs_pos)
    print(f"  done {time.time()-t0:.1f}s  peak={torch.cuda.max_memory_allocated(0)/1e9:.2f}GB",
          flush=True)
    torch.cuda.reset_peak_memory_stats(0)

    print(f"[forward] R'", flush=True)
    t0 = time.time()
    Rp_di, Rp_do = run_forward(model, layers,
                                torch.tensor([seq_Rp], dtype=torch.long),
                                abs_pos)
    print(f"  done {time.time()-t0:.1f}s  peak={torch.cuda.max_memory_allocated(0)/1e9:.2f}GB",
          flush=True)

    # ---- compute diff (down_in is per-layer list, dim varies; down_out is stacked) ----
    diff_di_list = [Rp_di[L] - R_di[L] for L in range(n_layers)]   # list of [n_pos, intermediate_L]
    diff_do = Rp_do - R_do     # [n_layers, n_pos, 1536]

    int_dims = [t.shape[-1] for t in diff_di_list]
    print(f"\n[diff] per-layer intermediate dims = {int_dims}", flush=True)
    print(f"       down_proj_out shape           = {tuple(diff_do.shape)}", flush=True)

    # ---- sanity: pos < 7 should be ≈ 0 ----
    print(f"\n[sanity] diff norm by gen-position (summed over layers & neurons)",
          flush=True)
    print(f"  pos | sum|diff_int| | sum|diff_down_out| | comment", flush=True)
    for i, gp in enumerate(GEN_POS_OF_INTEREST):
        s_di = sum(d[i].abs().sum().item() for d in diff_di_list)
        s_do = diff_do[:, i, :].abs().sum().item()
        if gp < REPLACE_POS[0]:
            note = "(pre-replace, expect 0)"
        elif gp in REPLACE_POS:
            note = "(REPLACEMENT)"
        else:
            note = "(post-replace, diffusion)"
        print(f"  {gp:>3} | {s_di:>13.2f} | {s_do:>17.2f} | {note}", flush=True)

    # ---- per-layer max|diff| over all positions ----
    # focus on positions 7+ (the actually different region)
    post_idx = [i for i, gp in enumerate(GEN_POS_OF_INTEREST) if gp >= REPLACE_POS[0]]
    do_post = diff_do[:, post_idx, :]    # [n_layers, n_post_pos, 1536]

    print(f"\n[layer-wise] over post-replacement positions (gen ≥ 7)", flush=True)
    print(f"  L  | dim   | max|diff_int| | mean|diff_int| | max|diff_do| | mean|diff_do|", flush=True)
    layer_stats = []
    for L in range(n_layers):
        di_post_L = diff_di_list[L][post_idx]   # [n_post_pos, intermediate_L]
        max_di = di_post_L.abs().max().item()
        mean_di = di_post_L.abs().mean().item()
        max_do = do_post[L].abs().max().item()
        mean_do = do_post[L].abs().mean().item()
        layer_stats.append((L, int_dims[L], max_di, mean_di, max_do, mean_do))
        print(f"  L{L:>2} | {int_dims[L]:>5} | {max_di:>13.4f} | {mean_di:>14.5f} | {max_do:>12.4f} | {mean_do:>13.5f}",
              flush=True)

    # ---- top neurons per layer (intermediate, gen pos 7 only — replacement onset) ----
    target_gp = 7
    target_idx_in_pos = GEN_POS_OF_INTEREST.index(target_gp)
    print(f"\n[top-{TOP_K}] per layer at gen-pos {target_gp} (replacement onset, intermediate)",
          flush=True)
    per_layer_top = {}
    for L in range(n_layers):
        d = diff_di_list[L][target_idx_in_pos, :]   # [intermediate_L]
        topv, topi = d.abs().topk(TOP_K)
        rows = []
        for v_abs, idx in zip(topv.tolist(), topi.tolist()):
            rows.append({
                "neuron": idx,
                "diff": d[idx].item(),
                "R_val":  R_di[L][target_idx_in_pos, idx].item(),
                "Rp_val": Rp_di[L][target_idx_in_pos, idx].item(),
            })
        per_layer_top[L] = rows

    # show top-3 per layer condensed
    print(f"  L  |   top-1            top-2            top-3", flush=True)
    for L in range(n_layers):
        rs = per_layer_top[L][:3]
        s = "  ".join(f"L{L}#{r['neuron']:<5d} d={r['diff']:+6.2f}" for r in rs)
        print(f"  L{L:>2} | {s}", flush=True)

    # ---- inventory cross-reference ----
    inventory = {}
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        if "known_neurons" in nj:
            inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "prompt": PROMPT_TEXT,
            "R_gen_ids": R_GEN_IDS,
            "Rp_replace_ids": R_PRIME_REPLACE,
            "replace_pos": REPLACE_POS,
            "gen_pos_of_interest": GEN_POS_OF_INTEREST,
            "abs_pos": abs_pos,
            "assistant_start": assistant_start,
            "intermediate_dims_per_layer": int_dims,
        },
        "R_down_in":  R_di,            # list[Tensor] per-layer
        "R_down_out": R_do,            # [L, n_pos, 1536]
        "Rp_down_in":  Rp_di,          # list[Tensor]
        "Rp_down_out": Rp_do,
        "diff_down_in_list": diff_di_list,   # list[Tensor]
        "diff_down_out": diff_do,
        "per_layer_top_at_pos7": per_layer_top,
    }, OUT_DIR / "step3_diff_raw.pt")
    print(f"\n[save] step3_diff_raw.pt", flush=True)

    # ---- text report ----
    with open(OUT_DIR / "step3_diff_report.txt", "w", encoding="utf-8") as f:
        f.write("================ Step 3: identity-knowledge differential ================\n\n")
        f.write(f"prompt              : {PROMPT_TEXT!r}\n")
        f.write(f"R   gen tokens (19) : Gemma greedy answer (Google DeepMind)\n")
        f.write(f"R'  gen tokens (19) : pos 7-9 replaced  ' Google Deep Mind' → ' Apple AI Lab'\n")
        f.write(f"replacement ids     : {R_PRIME_REPLACE}\n")
        f.write(f"assistant_start     : {assistant_start}\n")
        f.write(f"abs positions cap'd : {abs_pos}\n\n")

        f.write(f"--- token side-by-side ---\n")
        f.write(f"  pos | R                       | R'\n")
        for i in range(len(R_GEN_IDS)):
            a = seq_R[assistant_start + i]; b = seq_Rp[assistant_start + i]
            ta = repr(tok.decode([a], skip_special_tokens=False))
            tb = repr(tok.decode([b], skip_special_tokens=False))
            m = "==" if a == b else "≠≠"
            f.write(f"  {i:>3} | {a:>7} {ta:<22} | {b:>7} {tb:<22} | {m}\n")

        f.write(f"\n--- diff norm by gen-position ---\n")
        f.write(f"  pos | sum|diff_int| | sum|diff_down_out| | note\n")
        for i, gp in enumerate(GEN_POS_OF_INTEREST):
            s_di = sum(d[i].abs().sum().item() for d in diff_di_list)
            s_do = diff_do[:, i, :].abs().sum().item()
            if gp < REPLACE_POS[0]:
                note = "(pre-replace, sanity)"
            elif gp in REPLACE_POS:
                note = "(REPLACEMENT)"
            else:
                note = "(post-replace)"
            f.write(f"  {gp:>3} | {s_di:>13.2f} | {s_do:>17.2f} | {note}\n")

        f.write(f"\n--- per-layer max|diff| over gen pos ≥ 7 ---\n")
        f.write(f"  L  | dim   | max|diff_int| mean|diff_int| max|diff_do|  mean|diff_do|\n")
        for L, dim, mxi, mni, mxo, mno in layer_stats:
            f.write(f"  L{L:>2} | {dim:>5} | {mxi:>13.4f} {mni:>14.5f} {mxo:>12.4f}  {mno:>13.5f}\n")

        # rank layers by max|diff_int|
        sorted_L = sorted(layer_stats, key=lambda r: r[2], reverse=True)
        f.write(f"\n  Top-10 layers by max|diff_int|:\n")
        for L, dim, mxi, mni, mxo, mno in sorted_L[:10]:
            f.write(f"    L{L:>2}  dim={dim}  max_int={mxi:.4f}  mean_int={mni:.5f}\n")

        # per-layer top-K neurons at gen pos 7
        f.write(f"\n--- per-layer top-{TOP_K} neurons at gen pos 7 ('Google'/'Apple') ---\n")
        for L in range(n_layers):
            f.write(f"\n  L{L} top-{TOP_K}:\n")
            for i, r in enumerate(per_layer_top[L]):
                inv = inventory.get((L, r["neuron"]))
                tag = ""
                if inv:
                    tag = f"  [inv: {inv.get('tier','?')}, {inv.get('label','')[:50]}]"
                f.write(f"    {i+1:>3}. L{L}#{r['neuron']:<5d}  diff={r['diff']:+8.3f}  "
                        f"R={r['R_val']:+7.3f}  R'={r['Rp_val']:+7.3f}{tag}\n")

        f.write(f"\n--- inventory matches (across all layers, top-{TOP_K} per layer) ---\n")
        if not inventory:
            f.write("  (no neurons.json found)\n")
        else:
            n_match_total = 0
            for L in range(n_layers):
                matches = [r for r in per_layer_top[L]
                           if (L, r["neuron"]) in inventory]
                if matches:
                    f.write(f"\n  L{L}: {len(matches)} match(es)\n")
                    for r in matches:
                        inv = inventory[(L, r["neuron"])]
                        f.write(f"    L{L}#{r['neuron']:<5d}  diff={r['diff']:+7.3f}  "
                                f"[{inv.get('tier','?')}]  {inv.get('label','')}\n")
                    n_match_total += len(matches)
            f.write(f"\n  total matches: {n_match_total}\n")

    print(f"[save] step3_diff_report.txt", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
