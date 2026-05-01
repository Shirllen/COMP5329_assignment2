from __future__ import annotations

import argparse

from src.model import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on SST-2.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument(
        "--checkpoint-dir",
        "--output-dir",
        dest="checkpoint_dir",
        default="results/checkpoints/distilbert_sst2",
    )
    parser.add_argument("--metrics-dir", default="results/metrics/distilbert_sst2")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--metric-for-best-model", default="accuracy")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        TrainConfig(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            output_dir=args.checkpoint_dir,
            metrics_dir=args.metrics_dir,
            max_length=args.max_length,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            metric_for_best_model=args.metric_for_best_model,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
        )
    )


if __name__ == "__main__":
    main()
