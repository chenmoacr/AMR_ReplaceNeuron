"""
Phase 1: 验证 Opus cot 能否引导 Gemma4 续写出正确 C++ code.

输入:  [leetcode 233 user prompt] + soc + "thought\n" + Opus cot + "\n" + eoc
模型: 在 eoc 后续写 response (C++ code)
验证: manual trace n=13, n=0, n=824 期望值
"""
import os, sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")
QA_PATH = ROOT / "leetcode233" / "data" / "gemma_code_GB01.json"
OPUS_COT_PATH = ROOT / "leetcode233" / "data" / "opus_cot_leetcode233.txt"
OUT_PATH = ROOT / "leetcode233" / "upstream" / "phase1_opus_cot_output.txt"
DEVICE = "cuda:0"


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def main():
    import json
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = strip_user_block(data["A"][0]["query"])
    opus_cot = OPUS_COT_PATH.read_text(encoding="utf-8")

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

    soc = tok.convert_tokens_to_ids("<|channel>")
    eoc = tok.convert_tokens_to_ids("<channel|>")
    eot = tok.convert_tokens_to_ids("<turn|>")

    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()

    th_intro = tok.encode("thought\n", add_special_tokens=False)
    cot_body = tok.encode(opus_cot.strip(), add_special_tokens=False)
    nl = tok.encode("\n", add_special_tokens=False)

    seq = list(prompt_ids)
    seq.append(soc)
    seq.extend(th_intro)
    seq.extend(cot_body)
    seq.extend(nl)
    seq.append(eoc)

    print(f"[seq] prompt={len(prompt_ids)}  cot_body={len(cot_body)}  total_prefill={len(seq)}")
    input_ids = torch.tensor([seq], device=DEVICE)
    attention_mask = torch.ones_like(input_ids)

    print("\n[gen] forwarding...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=1500, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=[eot],
        )
    gen_ids = out[0][input_ids.shape[1]:].cpu().tolist()
    while gen_ids and gen_ids[-1] == eot:
        gen_ids = gen_ids[:-1]
    response = tok.decode(gen_ids, skip_special_tokens=False)

    print("\n" + "=" * 70)
    print("Opus cot:")
    print("=" * 70)
    print(opus_cot)
    print("\n" + "=" * 70)
    print(f"Gemma response (len={len(gen_ids)} tokens):")
    print("=" * 70)
    print(response)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("================ Opus cot ================\n")
        f.write(opus_cot)
        f.write("\n\n================ Gemma response ================\n")
        f.write(response)
        f.write(f"\n\n[meta] prompt_tokens={len(prompt_ids)} cot_tokens={len(cot_body)} response_tokens={len(gen_ids)}\n")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
