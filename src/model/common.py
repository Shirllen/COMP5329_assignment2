from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import set_seed as transformers_set_seed

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DATASET_NAME = "glue"
DEFAULT_DATASET_CONFIG = "sst2"
DEFAULT_CACHE_DIR = ".cache/huggingface"
LABEL_ID_TO_NAME = {0: "negative", 1: "positive"}


@dataclass
class RunConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: str = DEFAULT_DATASET_CONFIG
    cache_dir: str = DEFAULT_CACHE_DIR
    max_length: int = 128
    seed: int = 42


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    transformers_set_seed(seed)


def compute_classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    return {"accuracy": float(accuracy), "f1": float(f1)}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)


def write_text(path: str | Path, text: str) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(text, encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_run_config(path: str | Path) -> Optional[RunConfig]:
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    raw = load_json(path_obj)
    payload = raw.get("run_config", raw)
    allowed = {field.name for field in RunConfig.__dataclass_fields__.values()}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return RunConfig(**filtered)


def dump_run_config(path: str | Path, run_config: RunConfig, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"run_config": asdict(run_config)}
    if extra:
        payload.update(extra)
    write_json(path, payload)
