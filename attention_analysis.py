from __future__ import annotations

import argparse

from src.model import run_attention_analysis
from src.model.config import AnalysisConfig, normalize_analysis_subset, parse_number_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run token-importance analysis with attention, gradient x input, and leave-one-out rankings."
    )
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/distilbert_sst2")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--metrics-dir", default="results/metrics/distilbert_sst2")
    parser.add_argument("--analysis-dir", default="results/analysis/distilbert_sst2")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--analysis-subset", choices=["correct_only", "all"], default=None)
    parser.add_argument("--only-correct", action="store_true", default=False)
    parser.add_argument("--include-incorrect", action="store_true", default=False)
    parser.add_argument("--top-k-values", default="1,2,3")
    parser.add_argument("--top-ratios", default="")
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--case-study-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_subset is not None:
        analysis_subset = args.analysis_subset
    elif args.include_incorrect:
        analysis_subset = "all"
    else:
        analysis_subset = "correct_only"

    run_attention_analysis(
        AnalysisConfig(
            checkpoint_dir=args.checkpoint_dir,
            cache_dir=args.cache_dir,
            metrics_dir=args.metrics_dir,
            analysis_dir=args.analysis_dir,
            max_length=args.max_length,
            seed=args.seed,
            sample_size=args.sample_size,
            analysis_subset=normalize_analysis_subset(analysis_subset),
            top_k_values=parse_number_list(args.top_k_values, int),
            top_ratios=parse_number_list(args.top_ratios, float),
            random_trials=args.random_trials,
            max_validation_examples=args.max_validation_examples,
            case_study_count=args.case_study_count,
        )
    )


if __name__ == "__main__":
    main()
