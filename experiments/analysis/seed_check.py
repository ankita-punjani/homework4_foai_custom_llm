"""Robustness check: retrain both corpora with extra seeds and keep only the eval summaries.

The graded experiments use SEED=42. A single seed cannot tell a real change from luck,
so this reruns the same pipeline (custom_llm.py, unmodified except SEED) with seeds
1, 2, 3 for each corpus. SEED also changes the 90/10 split, initialization and batches.
The extension files are moved aside for the starter runs and restored afterwards.

    .venv/bin/python experiments/analysis/seed_check.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "seed_check"
EXT, PARKED = ROOT / "corpus/extension", ROOT / ".parked_extension"


def run(seed, label):
    script = ROOT / f".seed_{seed}.py"
    source = (ROOT / "custom_llm.py").read_text()
    assert source.count("SEED, N_EMBD") == 1
    script.write_text(source.replace("SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = 42,",
                                     f"SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = {seed},"))
    before = set((ROOT / "llm_runs").glob("*")) if (ROOT / "llm_runs").exists() else set()
    try:
        subprocess.run([sys.executable, script.name], cwd=ROOT, check=True, capture_output=True)
    finally:
        script.unlink()
    new = sorted(set((ROOT / "llm_runs").glob("*")) - before)
    run_dir = next(p for p in new if p.is_dir())
    dest = OUT / f"{label}_seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    for stage in ("untrained", "final"):
        shutil.copy(run_dir / "language_evals" / stage / "eval_summary.json", dest / f"{stage}_eval_summary.json")
    shutil.copy(run_dir / "language_evals/final/eval_results.json", dest / "final_eval_results.json")
    shutil.copy(run_dir / "history.json", dest / "history.json")
    for p in new:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    summary = json.loads((dest / "final_eval_summary.json").read_text())
    return {g: summary["by_group"][g]["correct"] for g in summary["by_group"]} | {"total": summary["overall"]["correct"]}


def main():
    rows = {}
    shutil.move(EXT, PARKED)
    try:
        for seed in (1, 2, 3):
            rows[f"starter_seed{seed}"] = run(seed, "starter")
    finally:
        shutil.move(PARKED, EXT)
    for seed in (1, 2, 3):
        rows[f"expanded_seed{seed}"] = run(seed, "expanded")
    (OUT / "seed_check_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for name, row in rows.items():
        print(name, row)


if __name__ == "__main__":
    main()
