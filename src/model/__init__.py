"""Public API for the DistilBERT SST-2 training, evaluation, and attention analysis pipeline."""

from __future__ import annotations

import sys

from src.model.api import AnalysisResult, EvalResult, TrainResult, evaluate, run_attention_analysis, train
from src.model.config import DEFAULT_CHECKPOINT_DIR, AnalysisConfig, EvalConfig, TrainConfig


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

# Preserve the historical import path `src.model.experiment` without keeping a
# separate compatibility module file.
experiment = sys.modules[__name__]
sys.modules.setdefault(f"{__name__}.experiment", experiment)

__all__ = [
    "AnalysisConfig",
    "AnalysisArtifacts",
    "AnalysisResult",
    "BaselineAnalysisConfig",
    "BaselineEvaluationConfig",
    "BaselineTrainConfig",
    "EvalConfig",
    "EvaluationArtifacts",
    "EvalResult",
    "TrainConfig",
    "TrainArtifacts",
    "TrainResult",
    "evaluate",
    "evaluate_checkpoint",
    "experiment",
    "run_attention_analysis",
    "train",
    "train_baseline",
]
