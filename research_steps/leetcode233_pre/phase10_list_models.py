"""
Phase 10 step 1: list available models on internal OpenAI-compatible endpoint.
Endpoint: http://192.168.100.1:8317
"""
import os, sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import urllib.request
import urllib.error

ENDPOINT = "http://192.168.100.1:8317"
API_KEY = "your-api-key-1"

OUT_PATH = Path("J:/amr/amr_wtf/outputs/gemma_code_GB01/phase10_models.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def get(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{ENDPOINT}/v1/models"
    print(f"[GET] {url}")
    try:
        data = get(url, headers)
    except urllib.error.HTTPError as e:
        print(f"[HTTPError] {e.code} {e.reason}")
        body = e.read().decode("utf-8", errors="replace")
        print(f"  body: {body[:500]}")
        return
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")
        return

    models = data.get("data", data) if isinstance(data, dict) else data
    print(f"\n[total models] {len(models) if isinstance(models, list) else '??'}")
    if isinstance(models, list):
        # group by provider/prefix
        gemini = []
        others = []
        for m in models:
            mid = m.get("id", str(m)) if isinstance(m, dict) else str(m)
            if "gemini" in mid.lower():
                gemini.append(mid)
            else:
                others.append(mid)
        print(f"\n[Gemini group, {len(gemini)}]")
        for m in gemini:
            print(f"  {m}")
        print(f"\n[Other groups, {len(others)} (first 30)]")
        for m in others[:30]:
            print(f"  {m}")
        if len(others) > 30:
            print(f"  ... ({len(others)-30} more)")

    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
