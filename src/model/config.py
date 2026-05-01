from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, TypeVar

from src.model.common import DEFAULT_CACHE_DIR, DEFAULT_MODEL_NAME

DEFAULT_CHECKPOINT_DIR = "results/checkpoints/distilbert_sst2"
DEFAULT_METRICS_DIR = "results/metrics/distilbert_sst2"
DEFAULT_ANALYSIS_DIR = "results/analysis/distilbert_sst2"
TRAIN_METRICS_FILENAME = "train_metrics.json"
RUN_CONFIG_FILENAME = "run_config.json"

AnalysisSubset = str
T = TypeVar("T")


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    cache_dir: str = DEFAULT_CACHE_DIR
    output_dir: str = DEFAULT_CHECKPOINT_DIR
    metrics_dir: str = DEFAULT_METRICS_DIR
    max_length: int = 128
    seed: int = 42
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    logging_steps: int = 50
    save_total_limit: int = 2
    metric_for_best_model: str = "accuracy"
    max_train_examples: Optional[int] = None
    max_eval_examples: Optional[int] = None


@dataclass
class EvalConfig:
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    cache_dir: Optional[str] = None
    metrics_dir: str = DEFAULT_METRICS_DIR
    split: str = "validation"
    max_length: Optional[int] = None
    seed: Optional[int] = None
    per_device_eval_batch_size: int = 32
    max_eval_examples: Optional[int] = None


@dataclass
class AnalysisConfig:
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    cache_dir: Optional[str] = None
    metrics_dir: str = DEFAULT_METRICS_DIR
    analysis_dir: str = DEFAULT_ANALYSIS_DIR
    max_length: Optional[int] = None
    seed: Optional[int] = None
    sample_size: int = 200
    analysis_subset: AnalysisSubset = "correct_only"
    top_k_values: Sequence[int] = field(default_factory=lambda: [1, 2, 3])
    top_ratios: Sequence[float] = field(default_factory=list)
    random_trials: int = 5
    max_validation_examples: Optional[int] = None
    case_study_count: int = 5


def parse_number_list(raw_value: str, cast_fn: Callable[[str], T]) -> List[T]:
    if not raw_value.strip():
        return []
    return [cast_fn(item.strip()) for item in raw_value.split(",") if item.strip()]


def normalize_analysis_subset(analysis_subset: str) -> AnalysisSubset:
    normalized = analysis_subset.strip().lower()
    if normalized not in {"correct_only", "all"}:
        raise ValueError("analysis_subset must be one of {'correct_only', 'all'}.")
    return normalized
