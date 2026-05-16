"""
Phase 2: base (Gemma 自 cot+code, 错) vs opus (Opus cot + Gemma code, 对) 的 MLP 神经元差分

抓取窗口: 每个 sequence 的 eoc 位置 + 之后 K 个 token (post-cot transition).
diff: per (layer, neuron) signed mean(opus_window) - mean(base_window).
top |diff| 给出 "Opus cot 引导下被激活的神经元" = 走对路径的关键候选.
"""
import os, sys, json
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
OPUS_RESP_PATH = ROOT / "leetcode233" / "upstream" / "phase1_opus_cot_output.txt"
OUT_DIR = ROOT / "leetcode233" / "upstream"

DEVICE = "cuda:0"
CHUNK = 1024
WINDOW = 30   # eoc + 后 30 个 token (cot→正文 transition)
L_LO = 15
TOP_K = 50


def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def strip_response_prefix(resp: str) -> str:
    s = resp.lstrip("\n")
    for prefix in ("<|channel|>", "<|channel>", "<channel|>", "<channel>"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.lstrip("\n")


def build_seq(tokenizer, user_text, cot_text, resp_text, soc, eoc, eot):
    """返回 (full_ids tensor, eoc_pos_in_full)"""
    msgs = [{"role": "user", "content": user_text}]
    prompt_o = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()
    th_intro = tokenizer.encode("thought\n", add_special_tokens=False)
    cot_body = tokenizer.encode(cot_text.strip(), add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    resp_ids = tokenizer.encode(resp_text, add_special_tokens=False)

    seq = list(prompt_ids)
    seq.append(soc)
    seq.extend(th_intro)
    seq.extend(cot_body)
    seq.extend(nl)
    eoc_pos = len(seq)
    seq.append(eoc)
    seq.extend(resp_ids)
    seq.append(eot)
    return torch.tensor(seq, dtype=torch.long), eoc_pos


def forward_capture_window(model, layers, full_ids, win_lo, win_hi):
    """跑 full_ids forward, 抓所有层 mlp.down_proj 输入在 [win_lo, win_hi) 范围内的 mean activation.
    Returns: list of [hidden] tensors per layer.
    """
    T = full_ids.shape[0]
    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    sums = [torch.zeros(s, dtype=torch.float32) for s in sizes]
    counts = [0] * len(layers)
    chunk_offset = {"v": 0}

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            gs = chunk_offset["v"]
            ov_lo = max(gs, win_lo)
            ov_hi = min(gs + chunk_len, win_hi)
            if ov_lo >= ov_hi:
                return
            local_lo = ov_lo - gs
            local_hi = ov_hi - gs
            seg = x[0, local_lo:local_hi, :].to(torch.float32)
            sums[idx] += seg.sum(dim=0).cpu()
            counts[idx] += (local_hi - local_lo)
        return fn

    hooks = [layers[i].mlp.down_proj.register_forward_hook(make_hook(i)) for i in range(len(layers))]
    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + CHUNK, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            cache_pos = torch.arange(pos, end, device=DEVICE)
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=chunk_ids, past_key_values=past_kv,
                            use_cache=True, cache_position=cache_pos)
            past_kv = out.past_key_values
            pos = end
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    means = []
    for i in range(len(layers)):
        c = max(counts[i], 1)
        means.append(sums[i] / c)
    return means


def extract_opus_response(text):
    """从 phase1 output 文件提取 'Gemma response' 段."""
    marker = "Gemma response"
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError("no Gemma response marker in phase1 output")
    # skip to first \n then skip ===== separator line
    nl = text.find("\n", idx) + 1
    sep = text.find("\n", nl) + 1  # past "=====" line
    end_marker = "\n\n[meta]"
    end = text.find(end_marker)
    if end < 0:
        end = len(text)
    return text[sep:end].strip()


def main():
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    item_a = data["A"][0]
    user_text = strip_user_block(item_a["query"])
    base_cot = item_a["cot"].strip("\n")
    base_resp = strip_response_prefix(item_a["response"])

    opus_cot = OPUS_COT_PATH.read_text(encoding="utf-8")
    phase1_text = OPUS_RESP_PATH.read_text(encoding="utf-8")
    opus_resp = extract_opus_response(phase1_text)
    print(f"[opus_resp head] {opus_resp[:120]!r}")

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
    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]

    soc = tok.convert_tokens_to_ids("<|channel>")
    eoc = tok.convert_tokens_to_ids("<channel|>")
    eot = tok.convert_tokens_to_ids("<turn|>")

    # Build sequences
    base_seq, base_eoc = build_seq(tok, user_text, base_cot, base_resp, soc, eoc, eot)
    opus_seq, opus_eoc = build_seq(tok, user_text, opus_cot, opus_resp, soc, eoc, eot)
    print(f"[seq-base] T={base_seq.shape[0]} eoc@{base_eoc}  window=[{base_eoc},{base_eoc+WINDOW})")
    print(f"[seq-opus] T={opus_seq.shape[0]} eoc@{opus_eoc}  window=[{opus_eoc},{opus_eoc+WINDOW})")

    # Sanity: print first 30 tokens after eoc, decoded
    base_after = tok.decode(base_seq[base_eoc:base_eoc+WINDOW].tolist(),
                             skip_special_tokens=False)
    opus_after = tok.decode(opus_seq[opus_eoc:opus_eoc+WINDOW].tolist(),
                             skip_special_tokens=False)
    print(f"\n[base post-eoc {WINDOW}t]\n  {base_after!r}")
    print(f"\n[opus post-eoc {WINDOW}t]\n  {opus_after!r}")

    print("\n[fwd-base]...")
    import time
    t0 = time.time()
    base_means = forward_capture_window(model, layers, base_seq,
                                          base_eoc, base_eoc + WINDOW)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    print("[fwd-opus]...")
    t0 = time.time()
    opus_means = forward_capture_window(model, layers, opus_seq,
                                          opus_eoc, opus_eoc + WINDOW)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # diff
    print(f"\n{'='*70}")
    print(f"TOP {TOP_K} |opus - base| MLP neurons (L>={L_LO})")
    print(f"{'='*70}")
    pairs = []
    for li in range(L_LO, n_layers):
        b = base_means[li]
        o = opus_means[li]
        d = o - b
        for n in range(d.shape[0]):
            dv = d[n].item()
            pairs.append((abs(dv), li, n, dv, b[n].item(), o[n].item()))
    pairs.sort(reverse=True)
    print(f"  {'rank':>4} {'layer#nrn':>11} {'opus':>7} {'base':>7} {'diff':>7}")
    for k, (_, li, n, dv, bv, ov) in enumerate(pairs[:TOP_K]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {ov:+.3f}  {bv:+.3f}  {dv:+.3f}")

    save_path = OUT_DIR / "phase2_diff.pt"
    torch.save({
        "qa_id": "phase2_base_vs_opus",
        "n_layers": n_layers,
        "sizes": sizes,
        "window": WINDOW,
        "base_eoc": base_eoc,
        "opus_eoc": opus_eoc,
        "base_means": base_means,
        "opus_means": opus_means,
        "top_diff": [(li, n, dv) for (_, li, n, dv, _, _) in pairs[:TOP_K]],
    }, save_path)
    print(f"\n[save] {save_path}")

    # also save text report
    with open(OUT_DIR / "phase2_diff.txt", "w", encoding="utf-8") as f:
        f.write(f"================ Phase 2 base vs opus diff ================\n")
        f.write(f"window = {WINDOW} tokens after eoc\n")
        f.write(f"base eoc@{base_eoc}, opus eoc@{opus_eoc}\n\n")
        f.write(f"base post-eoc: {base_after!r}\n")
        f.write(f"opus post-eoc: {opus_after!r}\n\n")
        f.write(f"Top {TOP_K} |opus - base|:\n")
        f.write(f"  {'rank':>4} {'layer#nrn':>11} {'opus':>7} {'base':>7} {'diff':>7}\n")
        for k, (_, li, n, dv, bv, ov) in enumerate(pairs[:TOP_K]):
            f.write(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {ov:+.3f}  {bv:+.3f}  {dv:+.3f}\n")
    print(f"[save] {OUT_DIR / 'phase2_diff.txt'}")


if __name__ == "__main__":
    main()
