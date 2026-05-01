from __future__ import annotations

"""Backward-compatible wrappers for the refactored DistilBERT SST-2 pipeline."""

from src.model.api import (
    AnalysisResult,
    EvalResult,
    TrainResult,
    evaluate,
    run_attention_analysis,
    train,
)
from src.model.config import (
    DEFAULT_CHECKPOINT_DIR,
    AnalysisConfig,
    EvalConfig,
    TrainConfig,
    normalize_analysis_subset,
    parse_number_list,
)


class BaselineTrainConfig(TrainConfig):
    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR, **kwargs):
        if "output_dir" not in kwargs:
            kwargs["output_dir"] = checkpoint_dir
        super().__init__(**kwargs)


BaselineEvaluationConfig = EvalConfig
BaselineAnalysisConfig = AnalysisConfig

TrainArtifacts = TrainResult
EvaluationArtifacts = EvalResult
AnalysisArtifacts = AnalysisResult

train_baseline = train
evaluate_checkpoint = evaluate

__all__ = [
    "AnalysisArtifacts",
    "AnalysisConfig",
    "BaselineAnalysisConfig",
    "BaselineEvaluationConfig",
    "BaselineTrainConfig",
    "EvalConfig",
    "EvaluationArtifacts",
    "TrainArtifacts",
    "TrainConfig",
    "TrainResult",
    "AnalysisResult",
    "EvalResult",
    "evaluate",
    "evaluate_checkpoint",
    "normalize_analysis_subset",
    "parse_number_list",
    "run_attention_analysis",
    "train",
    "train_baseline",
]
