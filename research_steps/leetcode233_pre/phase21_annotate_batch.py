"""
Phase 21 step 3: 1000 inert candidate 批量标注 (50/批 × 20 批).

- 序列调用 (gemini-2.5-pro 内部 endpoint, 避免并发碰撞)
- 每批写一个 JSONL chunk file, 主 JSONL 增量 append
- 断点续传: 若 chunk_##.json 已存在则跳过该批
- 错误 retry 3 次, 失败的批写 error.log

输出:
  phase21_annotations.jsonl    — 全 1000 个 merged record (1行1个)
  phase21_annotations_full.json — 全 raw response + metadata
  phase21_chunks/chunk_XX.json — 每批 50 个 (断点用)
"""
import os, sys, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import urllib.request
import urllib.error

ENDPOINT = "http://192.168.100.1:8317"
API_KEY = "your-api-key-1"
MODEL = "gemini-2.5-pro"

DATA_PATH = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_inert_for_gemini.json")
OUT_DIR = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_chunks")
OUT_JSONL = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_annotations.jsonl")
OUT_FULL = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_annotations_full.json")
ERR_LOG = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_errors.log")

BATCH_SIZE = 50
RETRY = 3
RETRY_DELAY = 5


# 同 pilot SYSTEM_PROMPT
SYSTEM_PROMPT = """You are a mechanistic interpretability assistant analyzing low-activation neurons in Gemma 4 E2B (~2B params, 35 layers, hidden=2560, matryoshka MLP: L0-L14 intermediate=6144, L15-L34 intermediate=12288, total ~338K MLP neurons).

CONTEXT — what we are hunting and WHY:
We need ~200 "safe-to-overwrite" MLP neurons whose W_gate and W_down weights we can rewrite WITHOUT damaging the model's existing capabilities, in order to install new "knowledge-anchor" detector neurons (these will detect a specific cot context and emit a marker token). Each input neuron in this batch has VERY LOW activation in our reference forward pass (the leetcode-233 cot+response trace under K=30 anxiety-clamp) — its cot_max and resp_max activations are 5-20x below the model's median. So we already know it doesn't FIRE on this trace. But that alone isn't enough — its W_gate detection direction and W_down output direction might still encode coherent function used in OTHER contexts (multilingual, long-form, tool use, etc.). If they do, overwriting destroys that function.

YOUR TASK:
For each neuron, look at its W_gate-row → vocab projection (top-15 push + top-15 suppress, ASCII-clean tokens) and W_down-column → vocab projection (top-15 push + top-15 suppress). Judge whether these projections look:
  (a) INCOHERENT NOISE = an unrelated grab-bag of tokens with no theme (digits + foreign words + random English + punctuation all jumbled). Safe to overwrite — score 0-2.
  (b) WEAKLY THEMED = mild lean (e.g. several whitespace tokens, or several punctuation). Probably low-impact but not zero. Score 3-5.
  (c) CLEARLY FUNCTIONAL = obvious theme (e.g. "all gate-push are numeric digits 0-9" / "all down-push are German articles" / "all gate-push are quote/bracket tokens"). NOT safe — score 7-10.

CRITICAL anti-bias guidance — Gemma 4 has 337,920 MLP neurons; many are vestigial, dead-on-arrival, or post-training-pruned residuals. LLMs (including you) tend to OVER-ATTRIBUTE function — finding "themes" in random noise to please the asker. PUSH BACK against this. When in doubt between "no theme" and "weak theme", choose "no theme". When a list shows just 2-3 vaguely related tokens out of 15 total, that's NOISE, not a theme.

Decision matrix for safe_to_overwrite:
  true  if coherence_score ≤ 3 AND no individually-critical token in top-3 (e.g. EOS, EOT, key control tokens)
  false otherwise

Output STRICT JSON ARRAY of N objects (one per input neuron, IN INPUT ORDER):
[
  {
    "id": "L##XXX",
    "gate_theme": "<short phrase | 'incoherent'>",
    "down_theme": "<short phrase | 'incoherent'>",
    "coherence_score": <0-10 int>,
    "safe_to_overwrite": <true|false>,
    "repurpose_risk": "<low|medium|high>",
    "confidence": "<high|medium|low>",
    "notes": "<1 short sentence>"
  },
  ...
]

NO markdown fences. NO preamble. NO trailing prose. Just the JSON array."""


def build_user_prompt(neurons):
    lines = [f"Annotate the following {len(neurons)} low-activation candidate neurons. Goal: identify safe-to-overwrite ones. Output JSON array (one object per neuron, in input order).\n"]
    for n in neurons:
        gate_push = [x["tok"] for x in n["gate_vocab_top_push"]]
        gate_supp = [x["tok"] for x in n["gate_vocab_top_suppress"]]
        down_push = [x["tok"] for x in n["down_vocab_top_push"]]
        down_supp = [x["tok"] for x in n["down_vocab_top_suppress"]]
        lines.append(f"--- {n['id']} (rank #{n['rank']}, layer {n['layer']}, index {n['index']}) ---")
        lines.append(f"cot_max={n['cot_max']:.4f}  resp_max={n['resp_max']:.4f}  "
                     f"||W_gate||={n['W_gate_row_norm']:.2f}  ||W_down||={n['W_down_col_norm']:.2f}")
        lines.append(f"gate_vocab_top_push: {gate_push}")
        lines.append(f"gate_vocab_top_suppress: {gate_supp}")
        lines.append(f"down_vocab_top_push: {down_push}")
        lines.append(f"down_vocab_top_suppress: {down_supp}")
        lines.append("")
    return "\n".join(lines)


def call_chat(model, system, user, timeout=600):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.time() - t0


def parse_response(content):
    s = content.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl+1:]
        if s.endswith("```"):
            s = s[:-3].rstrip()
    parsed = json.loads(s)
    if isinstance(parsed, dict):
        for k in ("neurons", "annotations", "items", "data", "result", "results"):
            if k in parsed and isinstance(parsed[k], list):
                return parsed[k]
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if lists:
            return lists[0]
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def log_err(msg):
    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if ERR_LOG.exists():
        ERR_LOG.unlink()

    all_neurons = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    n_total = len(all_neurons)
    print(f"[loaded] {n_total} candidates, batch_size={BATCH_SIZE}")

    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[plan] {n_batches} batches of up to {BATCH_SIZE}")

    all_annotations = {}        # id → annotation
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_elapsed = 0.0

    t_global = time.time()
    for bi in range(n_batches):
        start = bi * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_total)
        chunk_path = OUT_DIR / f"chunk_{bi:02d}.json"

        # 断点续传
        if chunk_path.exists():
            try:
                chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
                if len(chunk_data["annotations"]) == (end - start):
                    print(f"[batch {bi+1:>2}/{n_batches}] SKIP (cache hit: {chunk_path.name})")
                    for ann in chunk_data["annotations"]:
                        all_annotations[ann.get("id")] = ann
                    total_usage["prompt_tokens"] += chunk_data.get("usage", {}).get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += chunk_data.get("usage", {}).get("completion_tokens", 0)
                    total_usage["total_tokens"] += chunk_data.get("usage", {}).get("total_tokens", 0)
                    continue
            except Exception:
                pass

        neurons = all_neurons[start:end]
        user_prompt = build_user_prompt(neurons)

        annotations = None
        last_err = None
        for attempt in range(1, RETRY + 1):
            t0 = time.time()
            try:
                resp, elapsed = call_chat(MODEL, SYSTEM_PROMPT, user_prompt)
                content = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {})
                annotations = parse_response(content)
                if annotations is None:
                    raise ValueError("no array found in response")
                if len(annotations) != len(neurons):
                    raise ValueError(f"count mismatch: got {len(annotations)}, expected {len(neurons)}")
                # success
                chunk_path.write_text(json.dumps({
                    "batch_idx": bi, "start": start, "end": end,
                    "model": MODEL, "elapsed_sec": elapsed, "usage": usage,
                    "annotations": annotations, "raw_content": content,
                }, indent=1, ensure_ascii=False), encoding="utf-8")
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
                total_elapsed += elapsed
                # 累计统计
                safe_in_batch = sum(1 for a in annotations if a.get("safe_to_overwrite"))
                avg_coh = sum(a.get("coherence_score", 0) for a in annotations) / len(annotations)
                eta_sec = (n_batches - bi - 1) * (time.time() - t_global) / max(bi + 1, 1)
                print(f"[batch {bi+1:>2}/{n_batches}]  n={len(neurons)}  {elapsed:.1f}s  "
                      f"safe={safe_in_batch}/{len(neurons)}  avg_coh={avg_coh:.1f}  "
                      f"toks={usage.get('total_tokens', 0)}  "
                      f"(eta {eta_sec/60:.1f}m)")
                for ann in annotations:
                    all_annotations[ann.get("id")] = ann
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:500]
                last_err = f"HTTPError {e.code} {e.reason}: {body}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            print(f"  [retry {attempt}/{RETRY}] {last_err[:200]}")
            time.sleep(RETRY_DELAY)
        else:
            log_err(f"batch {bi} ({start}-{end}) FAILED: {last_err}")
            print(f"[batch {bi+1}] FAILED after {RETRY} retries: {last_err}")
            continue

    elapsed_total = time.time() - t_global
    print(f"\n[done] {elapsed_total:.1f}s total  ({elapsed_total/60:.1f}m)")
    print(f"[usage] {total_usage}")

    # ============= 主 JSONL 输出 (merge per neuron) =============
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for n in all_neurons:
            ann = all_annotations.get(n["id"], {})
            merged = {
                "rank": n["rank"], "id": n["id"], "layer": n["layer"], "index": n["index"],
                "cot_max": n["cot_max"], "resp_max": n["resp_max"],
                "combined_max": n["combined_max"],
                "annotation": ann,
                "raw_gate_top_push": [x["tok"] for x in n["gate_vocab_top_push"][:8]],
                "raw_down_top_push": [x["tok"] for x in n["down_vocab_top_push"][:8]],
                "raw_down_top_suppress": [x["tok"] for x in n["down_vocab_top_suppress"][:8]],
            }
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
    print(f"[save] {OUT_JSONL}")

    OUT_FULL.write_text(json.dumps({
        "model": MODEL, "n_neurons": n_total, "n_batches": n_batches,
        "batch_size": BATCH_SIZE, "elapsed_sec": elapsed_total,
        "usage": total_usage,
        "annotation_count": len(all_annotations),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {OUT_FULL}")

    # ============= 总结统计 =============
    print(f"\n[summary]")
    if all_annotations:
        n_ann = len(all_annotations)
        safe = sum(1 for a in all_annotations.values() if a.get("safe_to_overwrite"))
        score_dist = [0] * 11
        risk_dist = {"low":0, "medium":0, "high":0, "?":0}
        for a in all_annotations.values():
            sc = a.get("coherence_score", -1)
            if 0 <= sc <= 10:
                score_dist[sc] += 1
            rk = a.get("repurpose_risk", "?")
            risk_dist[rk] = risk_dist.get(rk, 0) + 1
        print(f"  total annotated: {n_ann}/{n_total}")
        print(f"  safe_to_overwrite=True: {safe} ({100*safe/n_ann:.1f}%)")
        print(f"  coherence dist:")
        for s in range(11):
            bar = "█" * (score_dist[s] // 5)
            print(f"    {s:>2}: {score_dist[s]:>4} {bar}")
        print(f"  risk dist: {risk_dist}")
        # 按层分布 (仅 safe)
        safe_by_layer = {}
        for n in all_neurons:
            ann = all_annotations.get(n["id"])
            if ann and ann.get("safe_to_overwrite"):
                safe_by_layer[n["layer"]] = safe_by_layer.get(n["layer"], 0) + 1
        print(f"\n  safe candidates 按层:")
        for L in sorted(safe_by_layer.keys()):
            n_safe = safe_by_layer[L]
            bar = "█" * (n_safe // 3)
            print(f"    L{L:02d}: {n_safe:>4} {bar}")


if __name__ == "__main__":
    main()
