"""
Phase 10 step 2.5: 先发 2 个 neuron 给 Gemini Flash (with thinking) 测 prompt + JSON 格式.
如果输出干净, 后续 phase10_annotate_batch.py 用 pro 批量 30.
"""
import os, sys, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import urllib.request
import urllib.error

ENDPOINT = "http://192.168.100.1:8317"
API_KEY = "your-api-key-1"
MODEL_TEST = "gemini-3-flash-preview"

DATA_PATH = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase10_neurons_for_gemini.json")
OUT_PATH = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase10_annotate_test.json")


SYSTEM_PROMPT = """You are a mechanistic interpretability assistant analyzing neurons in the Gemma 4 E2B language model (~2B params, 35 layers, hidden=1536). The model fails leetcode 233 (Number of Digit One) because its RLHF training installed an "anxiety / verbose textbook explanation" routine: when the model encounters digit-counting problems, it routes into a verbose framing (discuss complexity, worry about overflow, split into many cases, use string DP) instead of writing the lean positional digit DP solution directly.

Previous research found 30 MLP neurons whose K=30 clamp (gain=±1.5 in down_proj input) successfully flips the cot framing. We hypothesize these are RLHF-installed "anxiety / verbose-routine" neurons. Your task is to annotate each neuron based on its W_gate (gate detection direction) and W_down (output push direction) vocabulary projections.

For each neuron, you receive:
- id, layer, index, phase2_diff, clamp_gain
- gate_vocab_top_push: top-15 ASCII tokens that the W_gate column's projection onto lm_head highlights — approximates "what concept the gate reads as detection signal"
- gate_vocab_top_suppress: bottom-15 (anti-direction)
- down_vocab_top_push: top-15 ASCII tokens of W_down column projection — approximates "what tokens this neuron pushes when activated (positive activation)"
- down_vocab_top_suppress: bottom-15

Output ONLY a JSON object per neuron, no markdown fences, no extra text. Schema:
{
  "id": "L##XXX",
  "trigger_condition": "concise English description of what context/concept fires this neuron, inferred from gate_vocab_top_push. If unclear, say 'unclear (noisy)'.",
  "output_effect": "concise English description of what this neuron pushes (or suppresses) when activated, inferred from down_vocab_top_push and down_vocab_top_suppress. Note that clamp_gain inverts activation.",
  "functional_category": "ONE of: digit_topic_modulator | overflow_anxiety | magnitude_attention | algorithm_complexity_detector | testing_meta | programming_concept | hedging_uncertainty | structural_punctuation | unclear_polysemantic | other",
  "rlhf_anxiety_score": 0-10 (integer; 10 = clearly an RLHF-installed verbose-routine neuron; 0 = clearly unrelated; 5 = ambiguous),
  "confidence": "high | medium | low",
  "notes": "1 short sentence with reasoning or caveats"
}"""


def build_user_prompt(neurons):
    lines = [f"Annotate the following {len(neurons)} neurons. Output a JSON array (one object per neuron, same order).\n"]
    for n in neurons:
        # Strip projection values, only keep tokens for brevity
        gate_push = [x["tok"] for x in n["gate_vocab_top_push"]]
        gate_supp = [x["tok"] for x in n["gate_vocab_top_suppress"]]
        down_push = [x["tok"] for x in n["down_vocab_top_push"]]
        down_supp = [x["tok"] for x in n["down_vocab_top_suppress"]]
        lines.append(f"--- {n['id']} (rank #{n['rank']}, layer {n['layer']}, index {n['index']}) ---")
        lines.append(f"phase2_diff = {n['phase2_diff']:+.3f}, clamp_gain = {n['clamp_gain']:+.1f}")
        lines.append(f"gate_vocab_top_push: {gate_push}")
        lines.append(f"gate_vocab_top_suppress: {gate_supp}")
        lines.append(f"down_vocab_top_push: {down_push}")
        lines.append(f"down_vocab_top_suppress: {down_supp}")
        lines.append("")
    return "\n".join(lines)


def call_chat(model, system, user, timeout=120):
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
    elapsed = time.time() - t0
    return data, elapsed


def main():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    test_neurons = data[:2]   # first 2 for test
    print(f"[test] {len(test_neurons)} neurons → {MODEL_TEST}")

    user_prompt = build_user_prompt(test_neurons)
    print(f"\n[user prompt preview]\n{user_prompt[:600]}\n...")

    print(f"\n[call] {MODEL_TEST}...")
    try:
        resp, elapsed = call_chat(MODEL_TEST, SYSTEM_PROMPT, user_prompt)
        print(f"[done] in {elapsed:.1f}s")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[HTTPError] {e.code} {e.reason}\n  body: {body[:1000]}")
        return
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")
        return

    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    print(f"[usage] {usage}")
    print(f"\n[content]\n{content}")

    # Try to parse JSON
    try:
        parsed = json.loads(content)
        print(f"\n[JSON parse OK] keys/type: {type(parsed).__name__}")
        if isinstance(parsed, dict):
            # might be wrapper {"neurons": [...]}
            for k, v in parsed.items():
                print(f"  key={k!r}: type={type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}")
        elif isinstance(parsed, list):
            print(f"  list with {len(parsed)} entries")
            for entry in parsed:
                print(f"    {json.dumps(entry, ensure_ascii=False)[:200]}")
    except Exception as e:
        print(f"\n[JSON parse FAIL] {e}")

    OUT_PATH.write_text(json.dumps({
        "model": MODEL_TEST,
        "elapsed_sec": elapsed,
        "usage": usage,
        "raw_content": content,
        "neurons_sent": test_neurons,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
