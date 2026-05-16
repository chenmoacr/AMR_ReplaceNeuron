"""
Phase 10 step 3: 批量标注 30 个 neurons.

升级 prompt 后用 gemini-3-pro-preview, 一次发 30 个, 输出 JSON array.
保存为 JSONL + 完整 JSON 备份.
"""
import os, sys, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import urllib.request
import urllib.error

ENDPOINT = "http://192.168.100.1:8317"
API_KEY = "your-api-key-1"
MODEL = "gemini-2.5-pro"

DATA_PATH = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase10_neurons_for_gemini.json")
OUT_JSONL = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase10_annotations.jsonl")
OUT_FULL = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase10_annotations_full.json")


SYSTEM_PROMPT = """You are a mechanistic interpretability assistant analyzing neurons in the Gemma 4 E2B language model (~2B params, 35 layers, hidden=1536). The model fails leetcode 233 (Number of Digit One) because its RLHF training installed an "anxiety / verbose textbook explanation" routine: when the model encounters digit-counting problems, it routes into a verbose framing (discuss complexity, worry about overflow, split into many cases, use string DP) instead of writing the lean positional digit DP solution directly.

Previous research found 30 MLP neurons whose K=30 clamp (gain=±1.5 in down_proj input) successfully flips the cot framing. We hypothesize these are RLHF-installed "anxiety / verbose-routine" neurons. Your task is to annotate each neuron based on its W_gate (gate detection direction) and W_down (output push direction) vocabulary projections.

CRITICAL semantic distinction for accurate annotation:
- gate_vocab_top_push = tokens whose lm_head direction aligns POSITIVELY with W_gate column → "what concept the gate's detector READS as input signal"
- gate_vocab_top_suppress = anti-direction (usually less informative)
- down_vocab_top_push = tokens that W_down's output PUSHES UP in logit when neuron has POSITIVE activation
- down_vocab_top_suppress = tokens that W_down's output PUSHES DOWN (SUPPRESSES) when neuron has POSITIVE activation
- Note: clamp_gain inverts activation (negative gain → suppresses positive activation → reverses both push and suppress)

So if down_top_push = ['random_word_1', 'random_word_2'] and down_top_suppress = ['digit', 'tens', 'hundreds'], the neuron's output_effect is "SUPPRESSES digit-related vocabulary when activated" (not "pushes digit").

For each neuron, output strict JSON object with this schema:
{
  "id": "L##XXX",
  "trigger_condition": "concise English description of context that fires this neuron, inferred from gate_vocab_top_push (or 'unclear' if noisy)",
  "output_effect": "what activation does — accurate distinction between push/suppress sides",
  "functional_category": "ONE of: digit_topic_modulator | overflow_anxiety | magnitude_attention | algorithm_complexity_detector | testing_meta | programming_concept | hedging_uncertainty | structural_punctuation | unclear_polysemantic | other",
  "rlhf_anxiety_score": 0-10 integer (10 = clearly RLHF verbose-routine, 0 = unrelated, 5 = ambiguous),
  "confidence": "high | medium | low",
  "notes": "1 short sentence reasoning"
}

Output ONLY a single JSON array of 30 objects, in input order. NO markdown fences, NO extra prose."""


def build_user_prompt(neurons):
    lines = [f"Annotate the following {len(neurons)} neurons. Output a JSON array (one object per neuron, in input order).\n"]
    for n in neurons:
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
    elapsed = time.time() - t0
    return data, elapsed


def main():
    neurons = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"[loaded] {len(neurons)} neurons")

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

    # strip markdown fences if present
    content_stripped = content.strip()
    if content_stripped.startswith("```"):
        # remove leading fence (```json or ```)
        first_nl = content_stripped.find("\n")
        if first_nl > 0:
            content_stripped = content_stripped[first_nl+1:]
        # remove trailing fence
        if content_stripped.endswith("```"):
            content_stripped = content_stripped[:-3].rstrip()
    try:
        parsed = json.loads(content_stripped)
        # might be {"neurons": [...]} or raw [...]
        if isinstance(parsed, dict):
            for k in ("neurons", "annotations", "items", "data", "result", "results"):
                if k in parsed and isinstance(parsed[k], list):
                    annotations = parsed[k]
                    break
            else:
                # Take first list value
                lists = [v for v in parsed.values() if isinstance(v, list)]
                annotations = lists[0] if lists else None
        elif isinstance(parsed, list):
            annotations = parsed
        else:
            annotations = None
        if annotations is None:
            raise ValueError("can't find array in response")
        print(f"[parsed] {len(annotations)} annotations")
    except Exception as e:
        print(f"[parse FAIL] {e}")
        print(f"  raw content head 500: {content[:500]}")
        # Still save raw
        OUT_FULL.write_text(json.dumps({
            "model": MODEL, "elapsed_sec": elapsed, "usage": usage,
            "raw_content": content, "parse_error": str(e),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # Verify alignment: annotation count == input neurons
    if len(annotations) != len(neurons):
        print(f"[WARN] count mismatch: got {len(annotations)} expected {len(neurons)}")

    # Save JSONL (one neuron per line, merged input data + annotation)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, n in enumerate(neurons):
            ann = annotations[i] if i < len(annotations) else {}
            merged = {
                "rank": n["rank"], "id": n["id"], "layer": n["layer"], "index": n["index"],
                "phase2_diff": n["phase2_diff"], "clamp_gain": n["clamp_gain"],
                "annotation": ann,
                "raw_gate_top_push": [x["tok"] for x in n["gate_vocab_top_push"][:8]],
                "raw_down_top_push": [x["tok"] for x in n["down_vocab_top_push"][:8]],
                "raw_down_top_suppress": [x["tok"] for x in n["down_vocab_top_suppress"][:8]],
            }
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
    print(f"[save jsonl] {OUT_JSONL}")

    # Save full backup
    OUT_FULL.write_text(json.dumps({
        "model": MODEL, "elapsed_sec": elapsed, "usage": usage,
        "raw_content": content,
        "annotations": annotations,
        "n_neurons": len(neurons),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save full] {OUT_FULL}")

    # quick summary
    print(f"\n[summary]")
    cat_counts = {}
    score_sum = 0
    for ann in annotations:
        cat = ann.get("functional_category", "?")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        score_sum += ann.get("rlhf_anxiety_score", 0)
    for cat, c in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {c}")
    print(f"  avg rlhf_anxiety_score: {score_sum/max(len(annotations),1):.2f}")


if __name__ == "__main__":
    main()
