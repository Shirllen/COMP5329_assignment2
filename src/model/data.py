from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src.model.common import DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_NAME


def get_local_dataset_snapshot_dir(cache_dir: Optional[str]) -> Path:
    base_dir = Path(cache_dir or ".cache/huggingface")
    return base_dir / "glue_sst2_saved"


def load_sst2_dataset(cache_dir: Optional[str] = None) -> DatasetDict:
    snapshot_dir = get_local_dataset_snapshot_dir(cache_dir)
    if snapshot_dir.exists():
        return load_from_disk(str(snapshot_dir))

    dataset = load_dataset(DEFAULT_DATASET_NAME, DEFAULT_DATASET_CONFIG, cache_dir=cache_dir)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(snapshot_dir))
    return dataset


def tokenize_sst2(dataset: DatasetDict | Dataset, tokenizer, max_length: int):
    def preprocess(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        preprocess,
        batched=True,
        desc="Tokenizing SST-2",
    )


def maybe_select_subset(dataset: Dataset, max_examples: Optional[int]) -> Dataset:
    if max_examples is None or max_examples >= len(dataset):
        return dataset
    return dataset.select(range(max_examples))
