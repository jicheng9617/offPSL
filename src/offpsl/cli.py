"""Train and query the two supported methods using portable NumPy files."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import OfflineData, preferences
from .learner import fit, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True, help="NPZ containing X, Y and optional lower/upper")
    train.add_argument("--method", choices=["offpsl", "stch-psl"], default="offpsl")
    train.add_argument("--config")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--preferences", help="NPY rows on the objective simplex")
    predict.add_argument("--count", type=int, default=101)
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--out", required=True)
    for sub in (train, predict):
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "train":
        output = Path(args.out)
        if output.exists() and any(output.iterdir()):
            parser.error("--out must be new or empty; use a new run directory")
        config = Config.from_json(args.config) if args.config else Config()
        model = fit(OfflineData.from_npz(args.data), config, args.method, args.seed, args.device)
        output.mkdir(parents=True, exist_ok=True)
        model.save(output / "model.pt")
        (output / "training.json").write_text(json.dumps(model.report, indent=2), encoding="utf-8")
    else:
        output = Path(args.out)
        if output.exists():
            parser.error("--out already exists; use a new filename")
        model = load(args.checkpoint, args.device)
        weights = np.load(args.preferences, allow_pickle=False) if args.preferences else preferences(
            args.count, model.m, args.seed, model.config.preference_floor)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            np.savez_compressed(handle, **model.predict(weights))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
