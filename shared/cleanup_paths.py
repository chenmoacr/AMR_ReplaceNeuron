"""
cleanup_paths.py - one-shot script: rewrite the hard-coded J:/amr/ paths
in the core production scripts (neko/runtime, leetcode233/upstream, training,
eval) into portable forms:

    ROOT       = Path(__file__).resolve().parents[2]  # AMR_ReplaceNeuron root
    MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")

And remap ROOT-relative artefact paths from the original layout
(amr_wtf/outputs/gemma_code_GB01/*) to the new layout
(leetcode233/{data,training,weights,eval,upstream}/*).

Run from anywhere:
    python shared/cleanup_paths.py
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]   # AMR_ReplaceNeuron/

TARGETS = [
    REPO / "neko/runtime/runtime.py",
    REPO / "neko/runtime/precompute_meow_signature.py",
    REPO / "leetcode233/upstream/phase1_opus_cot_verify.py",
    REPO / "leetcode233/upstream/phase2_base_vs_opus_diff.py",
    REPO / "leetcode233/training/phase28a_lora_sft.py",
    REPO / "leetcode233/training/phase28b_targeted_sft.py",
    REPO / "leetcode233/training/phase28c_thinking_sft.py",
    REPO / "leetcode233/training/phase28d_cot_active_sft.py",
    REPO / "leetcode233/eval/phase29_cross_problem.py",
    REPO / "leetcode233/eval/phase29b_digit_dp_robust.py",
    REPO / "leetcode233/eval/phase30_delta_inspect.py",
]

# (literal pattern, replacement) — order matters
PATH_REWRITES = [
    # MODEL_PATH constant — common to all
    ('MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"',
     'MODEL_PATH = os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it")'),
    # In runtime.py the model_path is read from a config dict; just default-string it
    ('self.config.get("model_path", "J:/amr/models/gemma-4-E2B-it")',
     'self.config.get("model_path", os.environ.get("GEMMA_PATH", "/path/to/gemma-4-E2B-it"))'),
    # ROOT constant
    ('ROOT = Path("J:/amr/amr_wtf")',
     'ROOT = Path(__file__).resolve().parents[2]'),
    # absolute J:/amr/amr_wtf path appearing inline (precompute_meow uses one)
    ('Path("J:/amr/amr_wtf/identity_swap/step30_samples.json")',
     'Path(__file__).resolve().parents[2] / "research_steps" / "neko_pre" / "step30_samples.json"'),
    # ROOT-relative remappings (longer prefixes first)
    ('ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"',
     'ROOT / "leetcode233" / "data" / "gemma_code_GB01.json"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase21_chunks"',
     'ROOT / "leetcode233" / "data" / "phase21_chunks"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase21_inert_for_gemini.json"',
     'ROOT / "leetcode233" / "data" / "phase21_inert_for_gemini.json"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase20_dead_scan.pt"',
     'ROOT / "leetcode233" / "data" / "phase20_dead_scan.pt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.pt"',
     'ROOT / "leetcode233" / "data" / "phase2_diff.pt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28a_lora_weights.pt"',
     'ROOT / "leetcode233" / "weights" / "phase28a_lora_weights.pt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28b_deltas.pt"',
     'ROOT / "leetcode233" / "weights" / "phase28b_deltas.pt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28c_deltas.pt"',
     'ROOT / "leetcode233" / "weights" / "phase28c_deltas.pt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28d_deltas.pt"',
     'ROOT / "leetcode233" / "weights" / "phase28d_deltas.pt"'),
    # training run *.txt output paths
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28a_lora_sft.txt"',
     'ROOT / "leetcode233" / "training" / "phase28a_run.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28b_targeted_sft.txt"',
     'ROOT / "leetcode233" / "training" / "phase28b_run.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28c_thinking_sft.txt"',
     'ROOT / "leetcode233" / "training" / "phase28c_run.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase28d_cot_active.txt"',
     'ROOT / "leetcode233" / "training" / "phase28d_run.txt"'),
    # upstream output paths
    ('ROOT / "opus_cot_leetcode233.txt"',
     'ROOT / "leetcode233" / "data" / "opus_cot_leetcode233.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase1_opus_cot_output.txt"',
     'ROOT / "leetcode233" / "upstream" / "phase1_opus_cot_output.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase2_diff.txt"',
     'ROOT / "leetcode233" / "upstream" / "phase2_diff.txt"'),
    ('OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"',
     'OUT_DIR = ROOT / "leetcode233" / "upstream"'),
    # eval outputs
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase29_cross_problem.txt"',
     'ROOT / "leetcode233" / "eval" / "phase29_cross_problem.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase29b_digit_dp_robust.txt"',
     'ROOT / "leetcode233" / "eval" / "phase29b_digit_dp_robust.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase30_delta_inspect.txt"',
     'ROOT / "leetcode233" / "eval" / "phase30_delta_inspect.txt"'),
    ('ROOT / "outputs" / "gemma_code_GB01" / "phase30_delta_inspect.json"',
     'ROOT / "leetcode233" / "eval" / "phase30_delta_inspect.json"'),
]


def needs_os_import(src: str) -> bool:
    return ('os.environ.get("GEMMA_PATH"' in src) and not re.search(r'^import os\b', src, re.M)


def ensure_os_import(src: str) -> str:
    """Make sure 'import os' is present near the top of the file."""
    if re.search(r'^import os\b', src, re.M):
        return src
    # Find first import line, insert before it
    m = re.search(r'^(import |from )', src, re.M)
    if m:
        ins = m.start()
        return src[:ins] + "import os\n" + src[ins:]
    # No import line found; prepend
    return "import os\n" + src


def rewrite(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    orig = text
    n = 0
    for old, new in PATH_REWRITES:
        before = text
        text = text.replace(old, new)
        if text != before:
            n += 1
    if needs_os_import(text):
        text = ensure_os_import(text)
    changed = text != orig
    if changed:
        path.write_text(text, encoding="utf-8")
    return {"path": str(path.relative_to(REPO)), "patterns_applied": n, "changed": changed}


def remaining_hardcodes(path: Path) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "J:/amr/" in line:
            out.append((i, line.strip()))
    return out


def main():
    print(f"Repo: {REPO}")
    print(f"Targets: {len(TARGETS)}")
    print()
    for t in TARGETS:
        if not t.exists():
            print(f"  MISSING  {t.relative_to(REPO)}")
            continue
        r = rewrite(t)
        rem = remaining_hardcodes(t)
        flag = "  OK " if not rem else " WARN"
        print(f"  {flag}  patterns={r['patterns_applied']:2d}  remaining_J:/amr/={len(rem):2d}  {r['path']}")
        for ln, line in rem:
            print(f"        L{ln}: {line[:80]}")


if __name__ == "__main__":
    main()
