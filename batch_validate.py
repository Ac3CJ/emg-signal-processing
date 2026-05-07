"""
batch_validate.py
Run ModelValidator --validate_all for every trained model under neural-network-models/.

Usage:
    python batch_validate.py              # validate all discovered models
    python batch_validate.py step_20ms aug_all   # validate specific run names only

Output: console progress + full log written to validation-batch-logs/batch_validate_<timestamp>.log

To run unattended on Windows and keep output after closing the terminal:
    start /B python batch_validate.py > batch_run.log 2>&1
"""

import json
import os
import subprocess
import sys
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS_ROOT = "./neural-network-models"
VALIDATOR   = "ModelValidator.py"
LOG_DIR     = "./validation-batch-logs"

# Known arch run-name → model_type mappings.
# Anything not listed here defaults to 'rcnn'.
ARCH_MAP = {
    "arch_rcnn":      "rcnn",
    "arch_rnn":       "rnn",
    "arch_lstm":      "lstm",
    "arch_1dcnn":     "1dcnn",
    "arch_naive_ann": "naive_ann",
    "arch_naive_ann2":"naive_ann",
    "arch-naive-ann": "naive_ann",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_arch(run_name: str, training_dir: str) -> str:
    config_path = os.path.join(training_dir, "best_shoulder_rcnn_run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f).get("model_type", "rcnn")
    return ARCH_MAP.get(run_name, "rcnn")


def discover_runs(models_root: str) -> list[tuple[str, str]]:
    """Return (run_name, training_dir) for every dir that has a trained model."""
    runs = []
    for entry in sorted(os.listdir(models_root)):
        run_dir = os.path.join(models_root, entry)
        if not os.path.isdir(run_dir):
            continue
        training_dir = os.path.join(run_dir, "training")
        if os.path.exists(os.path.join(training_dir, "best_shoulder_rcnn.pth")):
            runs.append((entry, training_dir))
    return runs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    filter_names = set(sys.argv[1:])

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"batch_validate_{timestamp}.log")

    all_runs = discover_runs(MODELS_ROOT)
    if not all_runs:
        print(f"No trained models found under {MODELS_ROOT}/")
        sys.exit(1)

    runs = [(n, d) for n, d in all_runs if not filter_names or n in filter_names]
    if not runs:
        print(f"No runs matched: {filter_names}")
        sys.exit(1)

    print(f"Validating {len(runs)}/{len(all_runs)} model(s) — log: {log_path}\n")

    results = []
    with open(log_path, "w", encoding="utf-8") as log:

        def tee(msg: str):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        tee(f"Batch validation started: {timestamp}")
        tee("=" * 60 + "\n")

        for i, (run_name, training_dir) in enumerate(runs, 1):
            arch = _resolve_arch(run_name, training_dir)
            tee(f"[{i}/{len(runs)}] {run_name}  (arch={arch})")

            cmd = [
                sys.executable, VALIDATOR,
                "--name",         run_name,
                "--validate_all",
                "--arch",         arch,
            ]

            result = subprocess.run(cmd, stdout=log, stderr=log, text=True)

            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            tee(f"  -> {status}\n")
            results.append((run_name, arch, status))

        tee("=" * 60)
        tee("SUMMARY")
        tee("=" * 60)
        for run_name, arch, status in results:
            tee(f"  {run_name:<42} arch={arch:<12} {status}")

    print(f"\nFull log: {log_path}")


if __name__ == "__main__":
    main()
