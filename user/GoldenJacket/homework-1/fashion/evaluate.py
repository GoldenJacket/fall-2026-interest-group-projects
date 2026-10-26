"""Score available checkpoints with the starter's evaluator; report missing files explicitly.

    .venv/bin/python evaluate.py
"""

import json
from pathlib import Path

from fashion_fm.evaluate import evaluate_checkpoint

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[3] / "homework_1" / "fashion-fm"
CHECKPOINTS = {
    "mine_25_epochs_mps": HERE / "outputs" / "fashion-fm" / "latest.pt",
    "mine_25_epochs_pace_a40": HERE / "pace" / "latest.pt",  # PACE ICE job 6366980, copied back with scp
    "provided_fashion_fm_best": STARTER / "models" / "fashion-fm-best.pt",
}

def main():
    summary = {}
    for name, ckpt in CHECKPOINTS.items():
        if not ckpt.is_file():
            print(f"Skipping {name}: checkpoint not found at {ckpt}")
            continue
        report = evaluate_checkpoint(
            ckpt, HERE / "eval" / name, data_dir=HERE / "data",
            classifier_checkpoint=HERE / "eval" / "fashion-classifier.pt",  # one shared judge for both
            seed=2026, workers=0,  # macOS spawns workers by re-importing; in-process loading is simpler
        )
        summary[name] = report
        print(name, json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}, indent=1))
    if not summary:
        raise FileNotFoundError("No checkpoint files found; train a model or restore a checkpoint before evaluation.")
    output = HERE / "eval" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Evaluated {len(summary)} available checkpoint(s). Missing models are excluded from this summary.")


if __name__ == "__main__":
    main()
