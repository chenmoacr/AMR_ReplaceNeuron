"""
Phase 5a (per Opus review): minimum-dose rank-one edit on single neuron L33#11786.

W_down[33][:, 11786] -= (1 + α) * proj * e_diff_n
where:
  e_diff = lm_head["0"] - lm_head["$"]    (Bug A direction)
  proj   = W_down 列 在 e_diff 方向的原始投影

α=0: 只抹掉投影 (neutral, neuron 不再 push '0' over '$')
α=1: 完整翻转 (neuron 反向 push '$' over '0')
α>1: super-anti push correct

实验设置:
  - 无 K=30 clamp, 无 inject, 完全裸跑 leetcode 233
  - α sweep [0.3, 0.6, 0.9, 1.2, 1.5] (Opus 推荐, step32 力度窗口经验)
  - 每个 α 都 restore W_down 然后重新 apply
"""
import os, sys, json, re
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"

DEVICE = "cuda:0"
TARGET_L = 33
TARGET_I = 11786
ALPHAS = [0.3, 0.6, 0.9, 1.2, 1.5]
MAX_NEW = 6000


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
    lm_head = model.lm_head
    eot = tok.convert_tokens_to_ids("<turn|>")

    # Bug A direction: '0' over '$'
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    dollar_id = tok.encode("$", add_special_tokens=False)[0]
    print(f"[tok] '0' id={zero_id}, '$' id={dollar_id}")
    e_zero = lm_head.weight[zero_id].to(torch.float32).cpu()
    e_dollar = lm_head.weight[dollar_id].to(torch.float32).cpu()
    e_diff = e_zero - e_dollar
    e_diff_n = F.normalize(e_diff, dim=0)
    print(f"[e_diff] ||e_diff||={e_diff.norm().item():.3f}  ||e_zero||={e_zero.norm().item():.3f}  ||e_dollar||={e_dollar.norm().item():.3f}")

    # backup original W_down column
    W_down = layers[TARGET_L].mlp.down_proj.weight.data   # [hidden, intermediate]
    col_orig = W_down[:, TARGET_I].detach().clone()
    proj_orig = (col_orig.to(torch.float32).cpu() @ e_diff_n).item()
    print(f"[L{TARGET_L}#{TARGET_I}] original W_down 列 在 e_diff 上投影 = {proj_orig:+.4f}")
    print(f"  (phase 4c attribution: act=+4.44, proj=+0.317 — 这里 proj 略低因为 e_diff_n normalized)")
    # vocab top-K of the original column (sanity: what tokens does this neuron promote?)
    proj_all = col_orig.to(torch.float32).cpu() @ lm_head.weight.to(torch.float32).cpu().T
    topk_pos = torch.topk(proj_all, 10)
    topk_neg = torch.topk(-proj_all, 10)
    print(f"\n[L{TARGET_L}#{TARGET_I} vocab top 10 push (W_down @ lm_head.T 最大)]")
    for v, i in zip(topk_pos.values.tolist(), topk_pos.indices.tolist()):
        print(f"    proj={v:+.4f}  {tok.decode([i], skip_special_tokens=False)!r}")
    print(f"\n[L{TARGET_L}#{TARGET_I} vocab top 10 suppress (-W_down @ lm_head.T 最大)]")
    for v, i in zip(topk_neg.values.tolist(), topk_neg.indices.tolist()):
        print(f"    proj={-v:+.4f}  {tok.decode([i], skip_special_tokens=False)!r}")

    msgs = [{"role": "user", "content": user_text}]
    chat_input = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attn = chat_input["attention_mask"].to(DEVICE)

    def free_gen():
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=[eot],
            )
        gen_ids = out[0][input_ids.shape[1]:].cpu().tolist()
        stopped = (gen_ids and gen_ids[-1] == eot)
        while gen_ids and gen_ids[-1] == eot:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=False), len(gen_ids), stopped

    def signals(text):
        cnt_zero = len(re.findall(r"Contribution is 0", text))
        cnt_dollar = len(re.findall(r"Contribution is \$P\$|Contribution is \$P", text))
        has_if_d_gt_1 = bool(re.search(r"if\s*\(\s*[Dd]\s*>\s*1\s*\)", text))
        has_else_d_gt_1 = bool(re.search(r"else\s+if\s*\(\s*[Dd]\s*>\s*1\s*\)", text))
        has_count_p = bool(re.search(r"count\s*\+=\s*P\b", text))
        # string DP markers (regression check)
        has_string_dp = bool(re.search(r"string repr|S\[i\]|positions \$i", text))
        return {
            "Contribution is 0": cnt_zero,
            "Contribution is $P$": cnt_dollar,
            "has if (D > 1)": has_if_d_gt_1,
            "has else if (D > 1)": has_else_d_gt_1,
            "has count += P": has_count_p,
            "has string DP regression": has_string_dp,
        }

    results = {}

    # baseline (no edit)
    print(f"\n{'='*70}\nBaseline (no edit)\n{'='*70}")
    W_down[:, TARGET_I] = col_orig
    text, ln, stp = free_gen()
    s = signals(text)
    print(f"  gen_len={ln} stopped={stp}")
    for k, v in s.items():
        print(f"  {k}: {v}")
    results["baseline"] = {"text": text, "len": ln, "stopped": stp, "signals": s}

    # α sweep
    for alpha in ALPHAS:
        # restore + apply edit
        W_down[:, TARGET_I] = col_orig
        col = col_orig.to(torch.float32).cpu()
        proj = (col @ e_diff_n).item()
        col_new = col - (1.0 + alpha) * proj * e_diff_n   # rank-one suppress + reverse
        W_down[:, TARGET_I] = col_new.to(W_down.dtype).to(W_down.device)
        new_proj = (col_new @ e_diff_n).item()
        print(f"\n{'='*70}\nα = {alpha}  (proj_orig={proj_orig:+.4f} → new={new_proj:+.4f})\n{'='*70}")
        text, ln, stp = free_gen()
        s = signals(text)
        print(f"  gen_len={ln} stopped={stp}")
        for k, v in s.items():
            print(f"  {k}: {v}")
        results[f"alpha={alpha}"] = {"text": text, "len": ln, "stopped": stp,
                                       "signals": s, "alpha": alpha,
                                       "proj_after_edit": new_proj}

    # restore
    W_down[:, TARGET_I] = col_orig

    # save
    save_path = OUT_DIR / "phase5a_single_edit_sweep.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"================ Phase 5a: L{TARGET_L}#{TARGET_I} single-edit α sweep ================\n\n")
        f.write(f"Bug A direction: 'lm_head[\"0\"] - lm_head[\"$\"]'\n")
        f.write(f"Original W_down 列 在 e_diff 投影: {proj_orig:+.4f}\n")
        f.write(f"L33#11786 top push: {[tok.decode([i], skip_special_tokens=False) for i in topk_pos.indices.tolist()]}\n\n")
        for name, r in results.items():
            f.write(f"\n\n========== {name} ==========\n")
            f.write(f"gen_len={r['len']} stopped={r['stopped']}\n")
            if 'proj_after_edit' in r:
                f.write(f"proj_after_edit={r['proj_after_edit']:+.4f}\n")
            f.write(f"signals:\n")
            for k, v in r['signals'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\n--- text ---\n{r['text']}\n")
    print(f"\n[save] {save_path}")


if __name__ == "__main__":
    main()
