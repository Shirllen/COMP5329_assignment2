from __future__ import annotations

import argparse

from src.model import EvalConfig, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved DistilBERT SST-2 checkpoint.")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/distilbert_sst2")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--metrics-dir", default="results/metrics/distilbert_sst2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        EvalConfig(
            checkpoint_dir=args.checkpoint_dir,
            cache_dir=args.cache_dir,
            metrics_dir=args.metrics_dir,
            split=args.split,
            max_length=args.max_length,
            seed=args.seed,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            max_eval_examples=args.max_eval_examples,
        )
    )


if __name__ == "__main__":
    main()
