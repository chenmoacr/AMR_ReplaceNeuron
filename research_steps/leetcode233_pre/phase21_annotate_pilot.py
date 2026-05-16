"""
Phase 21 step 2 (pilot): 前 30 个 inert candidate 喂 gemini-2.5-pro 标注.

任务变化 (相比 phase10 的 anxiety 标注):
  phase10  问: "这个神经元是不是 RLHF anxiety routine?"
  phase21  问: "这个神经元 W_gate + W_down 是不是 incoherent / 无清晰功能?
                能否安全 overwrite 来 repurpose 做知识注入 anchor?"

输出 schema:
  id, coherence_score (0=完全噪声, 10=清晰功能),
  gate_theme (or 'incoherent'), down_theme (or 'incoherent'),
  dominant_function (or 'no clear function'),
  safe_to_overwrite (bool), repurpose_risk (low/med/high),
  confidence, notes

注意: prompt 主动反 LLM "过度归因" 偏见 - 提醒 Gemma 4 有 337K 神经元, 大量
冷门/噪声 neuron 完全正常. 不确定时倾向 'incoherent'.
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
OUT_JSONL = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_pilot_annotations.jsonl")
OUT_FULL = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase21_pilot_full.json")

N_PILOT = 30


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


def main():
    all_neurons = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"[loaded] {len(all_neurons)} total candidates")
    neurons = all_neurons[:N_PILOT]
    print(f"[pilot] using first {len(neurons)} (coldest combined_max)")

    user_prompt = build_user_prompt(neurons)
    print(f"[user prompt] {len(user_prompt)} chars")
    print(f"\n[call] {MODEL}...")
    try:
        resp, elapsed = call_chat(MODEL, SYSTEM_PROMPT, user_prompt)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[HTTPError] {e.code} {e.reason}\n  body: {body[:1500]}")
        return
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")
        return

    print(f"[done] in {elapsed:.1f}s")
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    print(f"[usage] {usage}")

    annotations = parse_response(content)
    if annotations is None:
        print(f"[parse FAIL] raw head 800:\n{content[:800]}")
        OUT_FULL.write_text(json.dumps({
            "model": MODEL, "elapsed_sec": elapsed, "usage": usage,
            "raw_content": content, "parse_error": "no array found",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    print(f"[parsed] {len(annotations)} annotations")
    if len(annotations) != len(neurons):
        print(f"[WARN] count mismatch: got {len(annotations)} expected {len(neurons)}")

    # save jsonl
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, n in enumerate(neurons):
            ann = annotations[i] if i < len(annotations) else {}
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
        "model": MODEL, "elapsed_sec": elapsed, "usage": usage,
        "raw_content": content, "annotations": annotations,
        "n_neurons": len(neurons),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {OUT_FULL}")

    # summary
    print(f"\n[summary]")
    safe = 0
    score_dist = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    risk_dist = {"low":0, "medium":0, "high":0}
    for ann in annotations:
        if ann.get("safe_to_overwrite"):
            safe += 1
        sc = ann.get("coherence_score", -1)
        if 0 <= sc <= 10:
            score_dist[sc] += 1
        rk = ann.get("repurpose_risk", "?")
        if rk in risk_dist:
            risk_dist[rk] += 1
    print(f"  safe_to_overwrite=True: {safe}/{len(annotations)}")
    print(f"  coherence_score dist: {score_dist}")
    print(f"  risk dist: {risk_dist}")
    print(f"\n[sample annotations]")
    for i, ann in enumerate(annotations[:6]):
        n = neurons[i]
        gp = [x["tok"] for x in n["gate_vocab_top_push"][:5]]
        dp = [x["tok"] for x in n["down_vocab_top_push"][:5]]
        print(f"\n  {ann.get('id')}:")
        print(f"    gate_theme: {ann.get('gate_theme')}")
        print(f"    down_theme: {ann.get('down_theme')}")
        print(f"    coherence: {ann.get('coherence_score')}  safe: {ann.get('safe_to_overwrite')}  risk: {ann.get('repurpose_risk')}")
        print(f"    notes: {ann.get('notes')}")
        print(f"    [raw gate top-5]: {gp}")
        print(f"    [raw down top-5]: {dp}")


if __name__ == "__main__":
    main()
