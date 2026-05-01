"""Public API for the DistilBERT SST-2 training, evaluation, and attention analysis pipeline."""

from src.model.api import AnalysisResult, EvalResult, TrainResult, evaluate, run_attention_analysis, train
from src.model.config import AnalysisConfig, EvalConfig, TrainConfig

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "EvalConfig",
    "EvalResult",
    "TrainConfig",
    "TrainResult",
    "evaluate",
    "run_attention_analysis",
    "train",
]
