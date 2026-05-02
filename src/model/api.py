from __future__ import annotations

from itertools import combinations
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.model.attention import (
    build_deletion_batch,
    compute_spearman_rank_correlation,
    format_ranked_tokens,
    rank_tokens_by_attention,
    rank_tokens_by_gradient_x_input,
    rank_tokens_by_leave_one_out,
    run_inference,
    summarize_deletion,
)
from src.model.common import (
    LABEL_ID_TO_NAME,
    RunConfig,
    compute_classification_metrics,
    dump_run_config,
    ensure_dir,
    maybe_load_run_config,
    set_global_seed,
    softmax,
    write_json,
    write_text,
)
from src.model.config import (
    RUN_CONFIG_FILENAME,
    TRAIN_METRICS_FILENAME,
    AnalysisConfig,
    EvalConfig,
    TrainConfig,
    normalize_analysis_subset,
)
from src.model.data import load_sst2_dataset, maybe_select_subset, tokenize_sst2


@dataclass
class TrainResult:
    checkpoint_dir: Path
    metrics_dir: Path
    metrics_path: Path
    run_config_path: Path
    best_checkpoint_dir: Optional[Path]
    metrics: Dict[str, Any]


@dataclass
class EvalResult:
    checkpoint_dir: Path
    metrics_dir: Path
    predictions_path: Path
    metrics_path: Path
    metrics: Dict[str, Any]


@dataclass
class AnalysisResult:
    checkpoint_dir: Path
    analysis_dir: Path
    sampled_examples_path: Path
    attention_rankings_path: Path
    importance_rankings_path: Path
    ranking_similarity_records_path: Path
    ranking_similarity_summary_path: Path
    ranking_similarity_markdown_path: Path
    deletion_records_path: Path
    deletion_summary_path: Path
    deletion_summary_markdown_path: Path
    qualitative_cases_path: Path
    confidence_curve_path: Path
    analysis_metadata_path: Path


def _load_saved_runtime_config(metrics_dir: str | Path) -> RunConfig:
    config_path = Path(metrics_dir, RUN_CONFIG_FILENAME)
    return maybe_load_run_config(config_path) or RunConfig()


def _resolve_runtime_overrides(
    metrics_dir: str | Path,
    cache_dir: Optional[str],
    max_length: Optional[int],
    seed: Optional[int],
) -> Tuple[str, int, int]:
    saved_config = _load_saved_runtime_config(metrics_dir)
    resolved_cache_dir = cache_dir or saved_config.cache_dir
    resolved_max_length = max_length or saved_config.max_length
    resolved_seed = seed if seed is not None else saved_config.seed
    return resolved_cache_dir, resolved_max_length, resolved_seed


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_k_settings(
    num_candidate_tokens: int,
    top_k_values: Sequence[int],
    top_ratios: Sequence[float],
) -> List[Tuple[str, float, int]]:
    settings: List[Tuple[str, float, int]] = []
    seen = set()
    for k_value in top_k_values:
        actual_k = min(num_candidate_tokens, int(k_value))
        if actual_k <= 0:
            continue
        key = ("top_k", actual_k)
        if key in seen:
            continue
        seen.add(key)
        settings.append(("top_k", float(k_value), actual_k))
    for ratio in top_ratios:
        actual_k = min(num_candidate_tokens, max(1, int(math.ceil(num_candidate_tokens * ratio))))
        key = ("ratio", actual_k, float(ratio))
        if key in seen:
            continue
        seen.add(key)
        settings.append(("ratio", float(ratio), actual_k))
    return settings


def _average_random_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys_to_average = [
        "new_confidence",
        "confidence_drop",
        "delta_prob_negative",
        "delta_prob_positive",
        "gold_label_prob_drop",
        "target_prob_drop",
        "prediction_flip",
        "becomes_incorrect",
        "new_prob_negative",
        "new_prob_positive",
        "new_target_probability",
    ]
    averaged = dict(records[0])
    for key in keys_to_average:
        averaged[key] = float(sum(record[key] for record in records) / len(records))
    averaged["deleted_tokens"] = f"<average of {len(records)} random trials>"
    averaged["trial_count"] = len(records)
    return averaged


def _dataframe_to_markdown(dataframe: pd.DataFrame, float_columns: Sequence[str]) -> str:
    headers = list(dataframe.columns)

    def format_cell(column: str, value: Any) -> str:
        if pd.isna(value):
            return ""
        if column in float_columns:
            return f"{float(value):.4f}"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in dataframe.iterrows():
        cells = [format_cell(column, row[column]) for column in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _announce(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def train(config: TrainConfig) -> TrainResult:
    _announce(
        "Train",
        (
            f"Starting fine-tuning with model={config.model_name}, epochs={config.num_train_epochs}, "
            f"train_batch_size={config.per_device_train_batch_size}, eval_batch_size={config.per_device_eval_batch_size}, "
            f"seed={config.seed}"
        ),
    )
    set_global_seed(config.seed)

    checkpoint_dir = ensure_dir(config.output_dir)
    metrics_dir = ensure_dir(config.metrics_dir)
    run_config = RunConfig(
        model_name=config.model_name,
        cache_dir=config.cache_dir,
        max_length=config.max_length,
        seed=config.seed,
    )

    _announce("Train", f"Loading SST-2 dataset from cache_dir={config.cache_dir}")
    raw_datasets = load_sst2_dataset(cache_dir=config.cache_dir)
    _announce("Train", "Loading tokenizer and tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, use_fast=True)
    tokenized_datasets = tokenize_sst2(raw_datasets, tokenizer=tokenizer, max_length=config.max_length)
    train_dataset = maybe_select_subset(tokenized_datasets["train"], config.max_train_examples)
    eval_dataset = maybe_select_subset(tokenized_datasets["validation"], config.max_eval_examples)
    _announce(
        "Train",
        f"Prepared datasets: train_examples={len(train_dataset)}, validation_examples={len(eval_dataset)}",
    )

    _announce("Train", "Loading pretrained model")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        num_labels=2,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        return compute_classification_metrics(logits=logits, labels=labels)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        do_train=True,
        do_eval=True,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=True,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    _announce("Train", "Launching Trainer.train(); progress bar will update during optimization")
    train_output = trainer.train()
    _announce("Train", "Running final validation on the selected eval split")
    evaluation_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    _announce("Train", f"Saving checkpoint to {checkpoint_dir}")
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    metrics_payload = {
        "train_metrics": train_output.metrics,
        "eval_metrics": evaluation_metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),
        "log_history": trainer.state.log_history,
    }
    metrics_path = metrics_dir / TRAIN_METRICS_FILENAME
    write_json(metrics_path, metrics_payload)

    run_config_path = metrics_dir / RUN_CONFIG_FILENAME
    dump_run_config(
        run_config_path,
        run_config=run_config,
        extra={
            "paths": {
                "checkpoint_dir": str(checkpoint_dir),
                "metrics_dir": str(metrics_dir),
            },
            "training": {
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "num_train_epochs": config.num_train_epochs,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "per_device_eval_batch_size": config.per_device_eval_batch_size,
                "metric_for_best_model": config.metric_for_best_model,
                "logging_steps": config.logging_steps,
                "save_total_limit": config.save_total_limit,
                "max_train_examples": config.max_train_examples,
                "max_eval_examples": config.max_eval_examples,
            },
        },
    )

    best_checkpoint_dir = None
    if trainer.state.best_model_checkpoint:
        best_checkpoint_dir = Path(trainer.state.best_model_checkpoint)

    _announce(
        "Train",
        (
            f"Completed training. best_metric={trainer.state.best_metric}, "
            f"best_checkpoint={best_checkpoint_dir or checkpoint_dir}, metrics_path={metrics_path}"
        ),
    )
    return TrainResult(
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        metrics_path=metrics_path,
        run_config_path=run_config_path,
        best_checkpoint_dir=best_checkpoint_dir,
        metrics=metrics_payload,
    )


def evaluate(config: EvalConfig) -> EvalResult:
    _announce("Eval", f"Starting evaluation for checkpoint_dir={config.checkpoint_dir} on split={config.split}")
    cache_dir, max_length, seed = _resolve_runtime_overrides(
        metrics_dir=config.metrics_dir,
        cache_dir=config.cache_dir,
        max_length=config.max_length,
        seed=config.seed,
    )
    set_global_seed(seed)

    _announce("Eval", f"Loading SST-2 split with cache_dir={cache_dir}")
    raw_datasets = load_sst2_dataset(cache_dir=cache_dir)
    raw_split = maybe_select_subset(raw_datasets[config.split], config.max_eval_examples)

    _announce("Eval", "Loading tokenizer and checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint_dir)
    device = _resolve_device()
    model.to(device)
    model.eval()
    _announce("Eval", f"Using device={device}; num_examples={len(raw_split)}")

    _announce("Eval", "Tokenizing evaluation split")
    tokenized_split = tokenize_sst2(raw_split, tokenizer=tokenizer, max_length=max_length)
    tokenized_split.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(
        tokenized_split,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_logits = []
    all_labels = []
    losses = []
    progress_bar = tqdm(data_loader, total=len(data_loader), desc=f"Evaluating {config.split}", unit="batch")
    for batch_idx, batch in enumerate(progress_bar, start=1):
        labels = batch["labels"].to(device)
        model_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": labels,
        }
        with torch.no_grad():
            outputs = model(**model_inputs)
        all_logits.append(outputs.logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        losses.append(float(outputs.loss.detach().cpu().item()))
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(data_loader):
            progress_bar.set_postfix(avg_loss=f"{np.mean(losses):.4f}")

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)

    metrics = compute_classification_metrics(logits=logits, labels=labels)
    metrics["loss"] = float(np.mean(losses))
    metrics["num_examples"] = int(len(raw_split))

    predictions_df = pd.DataFrame(
        {
            f"{config.split}_index": np.arange(len(raw_split)),
            "sentence": raw_split["sentence"],
            "label": labels,
            "label_name": [LABEL_ID_TO_NAME[int(label)] for label in labels],
            "prediction": predictions,
            "prediction_name": [LABEL_ID_TO_NAME[int(pred)] for pred in predictions],
            "correct": predictions == labels,
            "prob_negative": probabilities[:, 0],
            "prob_positive": probabilities[:, 1],
            "confidence": confidence,
        }
    )

    metrics_dir = ensure_dir(config.metrics_dir)
    predictions_path = metrics_dir / f"{config.split}_predictions.csv"
    metrics_path = metrics_dir / f"{config.split}_metrics.json"
    predictions_df.to_csv(predictions_path, index=False)
    write_json(metrics_path, metrics)
    _announce(
        "Eval",
        (
            f"Completed evaluation. accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, "
            f"loss={metrics['loss']:.4f}, metrics_path={metrics_path}"
        ),
    )

    return EvalResult(
        checkpoint_dir=Path(config.checkpoint_dir),
        metrics_dir=metrics_dir,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )


def run_attention_analysis(config: AnalysisConfig) -> AnalysisResult:
    _announce(
        "Analysis",
        (
            f"Starting token-importance analysis with sample_size={config.sample_size}, "
            f"analysis_subset={config.analysis_subset}, top_k_values={list(config.top_k_values)}, "
            f"random_trials={config.random_trials}"
        ),
    )
    analysis_subset = normalize_analysis_subset(config.analysis_subset)
    cache_dir, max_length, seed = _resolve_runtime_overrides(
        metrics_dir=config.metrics_dir,
        cache_dir=config.cache_dir,
        max_length=config.max_length,
        seed=config.seed,
    )
    set_global_seed(seed)

    top_k_values = [int(value) for value in config.top_k_values]
    top_ratios = [float(value) for value in config.top_ratios]
    if not top_k_values and not top_ratios:
        raise ValueError("At least one deletion size must be provided via top_k_values or top_ratios.")

    analyze_only_correct_predictions = analysis_subset == "correct_only"
    analysis_dir = ensure_dir(config.analysis_dir)
    ranking_methods = ["attention", "grad_x_input", "loo"]

    _announce("Analysis", "Loading tokenizer and checkpoint for analysis")
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint_dir,
        attn_implementation="eager",
    )
    device = _resolve_device()
    model.to(device)
    model.eval()
    _announce("Analysis", f"Using device={device}")

    _announce("Analysis", f"Loading validation split from cache_dir={cache_dir}")
    raw_validation = load_sst2_dataset(cache_dir=cache_dir)["validation"]
    raw_validation = maybe_select_subset(raw_validation, config.max_validation_examples)
    _announce("Analysis", f"Scanning {len(raw_validation)} validation examples to build the candidate pool")

    candidate_examples = []
    filter_progress = tqdm(
        enumerate(raw_validation),
        total=len(raw_validation),
        desc="Filtering analysis candidates",
        unit="example",
    )
    for idx, example in filter_progress:
        encoded = tokenizer(
            example["sentence"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        base_output = run_inference(model, encoded, device=device, output_attentions=False)
        correct = int(base_output["prediction"] == int(example["label"]))
        if analyze_only_correct_predictions and not correct:
            continue
        candidate_examples.append(
            {
                "validation_index": idx,
                "sentence": example["sentence"],
                "label": int(example["label"]),
                "correct": correct,
            }
        )
        if idx == 0 or (idx + 1) % 50 == 0 or (idx + 1) == len(raw_validation):
            filter_progress.set_postfix(candidates=len(candidate_examples))

    if not candidate_examples:
        raise ValueError("No validation examples available for analysis after filtering.")

    rng = random.Random(seed)
    sample_size = min(config.sample_size, len(candidate_examples))
    sampled_examples = rng.sample(candidate_examples, sample_size)
    _announce(
        "Analysis",
        f"Candidate pool ready: {len(candidate_examples)} examples; sampled {len(sampled_examples)} for ranking and deletion analysis",
    )

    selection_rows = []
    attention_rows = []
    importance_rows = []
    similarity_rows = []
    deletion_rows = []
    case_rows = []

    sample_progress = tqdm(
        sampled_examples,
        total=len(sampled_examples),
        desc="Analyzing sampled examples",
        unit="sample",
    )
    for sample_idx, sample in enumerate(sample_progress, start=1):
        label = int(sample["label"])
        encoded = tokenizer(
            sample["sentence"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        base_output = run_inference(model, encoded, device=device, output_attentions=True)
        probabilities = base_output["probabilities"][0].numpy()
        target_label = int(base_output["prediction"])
        base_target_probability = float(probabilities[target_label])

        ranked_tokens_by_method = {
            "attention": rank_tokens_by_attention(tokenizer, encoded, base_output["attentions"]),
            "grad_x_input": rank_tokens_by_gradient_x_input(
                model,
                tokenizer,
                encoded,
                device=device,
                target_label=target_label,
            ),
            "loo": rank_tokens_by_leave_one_out(
                model,
                tokenizer,
                encoded,
                device=device,
                target_label=target_label,
                base_target_probability=base_target_probability,
                pad_token_id=tokenizer.pad_token_id,
            ),
        }
        attention_ranked = ranked_tokens_by_method["attention"]
        if not attention_ranked.positions:
            continue

        selection_rows.append(
            {
                "validation_index": sample["validation_index"],
                "sentence": sample["sentence"],
                "label": label,
                "label_name": LABEL_ID_TO_NAME[label],
                "base_prediction": target_label,
                "base_prediction_name": LABEL_ID_TO_NAME[target_label],
                "base_confidence": base_output["confidence"],
                "base_target_probability": base_target_probability,
                "correct": sample["correct"],
            }
        )

        for method_name, ranked_tokens in ranked_tokens_by_method.items():
            for rank, (position, token, score) in enumerate(
                zip(ranked_tokens.positions, ranked_tokens.tokens, ranked_tokens.scores),
                start=1,
            ):
                ranking_row = {
                    "validation_index": sample["validation_index"],
                    "method": method_name,
                    "rank": rank,
                    "token_position": position,
                    "token": token,
                    "importance_score": score,
                }
                importance_rows.append(ranking_row)
                if method_name == "attention":
                    attention_rows.append(
                        {
                            "validation_index": sample["validation_index"],
                            "rank": rank,
                            "token_position": position,
                            "token": token,
                            "attention_score": score,
                        }
                    )

        for method_a, method_b in combinations(ranking_methods, 2):
            similarity_rows.append(
                {
                    "validation_index": sample["validation_index"],
                    "method_a": method_a,
                    "method_b": method_b,
                    "ranked_token_count": len(ranked_tokens_by_method[method_a].positions),
                    "spearman_rank_correlation": compute_spearman_rank_correlation(
                        ranked_tokens_by_method[method_a],
                        ranked_tokens_by_method[method_b],
                    ),
                }
            )

        k_settings = _resolve_k_settings(len(attention_ranked.positions), top_k_values, top_ratios)
        primary_case_record: Optional[Dict[str, Any]] = None
        for setting_type, raw_k_value, actual_k in k_settings:
            if primary_case_record is None:
                primary_case_record = {
                    "validation_index": sample["validation_index"],
                    "sentence": sample["sentence"],
                    "label_name": LABEL_ID_TO_NAME[label],
                    "base_prediction_name": LABEL_ID_TO_NAME[target_label],
                    "base_prob_negative": float(probabilities[0]),
                    "base_prob_positive": float(probabilities[1]),
                    "case_setting_type": setting_type,
                    "case_setting_value": raw_k_value,
                    "case_actual_k": actual_k,
                }

            for method_name, ranked_tokens in ranked_tokens_by_method.items():
                selected_positions = ranked_tokens.positions[:actual_k]
                selected_tokens = ranked_tokens.tokens[:actual_k]
                deleted_batch = build_deletion_batch(
                    encoded_batch=encoded,
                    positions_to_delete=selected_positions,
                    pad_token_id=tokenizer.pad_token_id,
                )
                deleted_output = run_inference(model, deleted_batch, device=device, output_attentions=False)
                new_probabilities = deleted_output["probabilities"][0].numpy()
                deletion_summary = summarize_deletion(
                    original_probabilities=probabilities,
                    original_prediction=target_label,
                    new_probabilities=new_probabilities,
                    gold_label=label,
                )
                new_target_probability = float(new_probabilities[target_label])
                deletion_record = {
                    "method": method_name,
                    "validation_index": sample["validation_index"],
                    "sentence": sample["sentence"],
                    "label": label,
                    "label_name": LABEL_ID_TO_NAME[label],
                    "base_prediction": target_label,
                    "base_prediction_name": LABEL_ID_TO_NAME[target_label],
                    "target_label": target_label,
                    "target_label_name": LABEL_ID_TO_NAME[target_label],
                    "base_prob_negative": float(probabilities[0]),
                    "base_prob_positive": float(probabilities[1]),
                    "base_target_probability": base_target_probability,
                    "k_setting_type": setting_type,
                    "k_setting_value": raw_k_value,
                    "actual_k": actual_k,
                    "deleted_tokens": " ".join(selected_tokens),
                    "new_prob_negative": float(new_probabilities[0]),
                    "new_prob_positive": float(new_probabilities[1]),
                    "new_target_probability": new_target_probability,
                    "target_prob_drop": float(base_target_probability - new_target_probability),
                    "trial_count": 1,
                }
                deletion_record.update(deletion_summary)
                deletion_rows.append(deletion_record)

                if (
                    primary_case_record["case_setting_type"] == setting_type
                    and primary_case_record["case_actual_k"] == actual_k
                ):
                    primary_case_record[f"{method_name}_top_tokens"] = format_ranked_tokens(
                        ranked_tokens.tokens,
                        ranked_tokens.scores,
                        top_n=8,
                    )
                    primary_case_record[f"{method_name}_deleted_tokens"] = " ".join(selected_tokens)
                    primary_case_record[f"{method_name}_confidence_drop"] = deletion_record["confidence_drop"]
                    primary_case_record[f"{method_name}_target_prob_drop"] = deletion_record["target_prob_drop"]
                    primary_case_record[f"{method_name}_prediction_flip"] = deletion_record["prediction_flip"]
                    primary_case_record[f"{method_name}_new_prob_negative"] = deletion_record["new_prob_negative"]
                    primary_case_record[f"{method_name}_new_prob_positive"] = deletion_record["new_prob_positive"]

            if config.random_trials > 0:
                random_records = []
                for trial_idx in range(config.random_trials):
                    random_positions = rng.sample(attention_ranked.positions, actual_k)
                    random_tokens = [
                        tokenizer.convert_ids_to_tokens(encoded["input_ids"][0][position].item())
                        for position in random_positions
                    ]
                    random_batch = build_deletion_batch(
                        encoded_batch=encoded,
                        positions_to_delete=random_positions,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    random_output = run_inference(model, random_batch, device=device, output_attentions=False)
                    random_summary = summarize_deletion(
                        original_probabilities=probabilities,
                        original_prediction=target_label,
                        new_probabilities=random_output["probabilities"][0].numpy(),
                        gold_label=label,
                    )
                    new_target_probability = float(random_output["probabilities"][0][target_label].item())
                    random_record = {
                        "method": "random",
                        "validation_index": sample["validation_index"],
                        "sentence": sample["sentence"],
                        "label": label,
                        "label_name": LABEL_ID_TO_NAME[label],
                        "base_prediction": target_label,
                        "base_prediction_name": LABEL_ID_TO_NAME[target_label],
                        "target_label": target_label,
                        "target_label_name": LABEL_ID_TO_NAME[target_label],
                        "base_prob_negative": float(probabilities[0]),
                        "base_prob_positive": float(probabilities[1]),
                        "base_target_probability": base_target_probability,
                        "k_setting_type": setting_type,
                        "k_setting_value": raw_k_value,
                        "actual_k": actual_k,
                        "deleted_tokens": " ".join(random_tokens),
                        "new_prob_negative": float(random_output["probabilities"][0][0].item()),
                        "new_prob_positive": float(random_output["probabilities"][0][1].item()),
                        "new_target_probability": new_target_probability,
                        "target_prob_drop": float(base_target_probability - new_target_probability),
                        "trial_index": trial_idx,
                        "trial_count": 1,
                    }
                    random_record.update(random_summary)
                    random_records.append(random_record)
                deletion_rows.append(_average_random_records(random_records))

        if primary_case_record is not None:
            case_rows.append(primary_case_record)
        if sample_idx == 1 or sample_idx % 10 == 0 or sample_idx == len(sampled_examples):
            sample_progress.set_postfix(records=len(deletion_rows))

    if not deletion_rows:
        raise ValueError("Deletion analysis produced no records.")

    selection_df = pd.DataFrame(selection_rows).sort_values("validation_index")
    attention_df = pd.DataFrame(attention_rows).sort_values(["validation_index", "rank"])
    importance_df = pd.DataFrame(importance_rows).sort_values(["validation_index", "method", "rank"])
    similarity_df = pd.DataFrame(similarity_rows).sort_values(["method_a", "method_b", "validation_index"])
    deletion_df = pd.DataFrame(deletion_rows)

    similarity_summary_df = (
        similarity_df.groupby(["method_a", "method_b"], dropna=False)
        .agg(
            num_examples=("validation_index", "count"),
            mean_ranked_token_count=("ranked_token_count", "mean"),
            mean_spearman_rank_correlation=("spearman_rank_correlation", "mean"),
            std_spearman_rank_correlation=("spearman_rank_correlation", "std"),
        )
        .reset_index()
    )
    similarity_summary_df["std_spearman_rank_correlation"] = similarity_summary_df[
        "std_spearman_rank_correlation"
    ].fillna(0.0)

    summary_df = (
        deletion_df.groupby(["method", "k_setting_type", "k_setting_value", "actual_k"], dropna=False)
        .agg(
            num_examples=("validation_index", "count"),
            mean_confidence_drop=("confidence_drop", "mean"),
            std_confidence_drop=("confidence_drop", "std"),
            mean_gold_label_prob_drop=("gold_label_prob_drop", "mean"),
            mean_target_prob_drop=("target_prob_drop", "mean"),
            flip_rate=("prediction_flip", "mean"),
            becomes_incorrect_rate=("becomes_incorrect", "mean"),
            mean_delta_prob_negative=("delta_prob_negative", "mean"),
            mean_delta_prob_positive=("delta_prob_positive", "mean"),
        )
        .reset_index()
    )
    summary_df["std_confidence_drop"] = summary_df["std_confidence_drop"].fillna(0.0)

    sampled_examples_path = analysis_dir / "sampled_validation_examples.csv"
    attention_rankings_path = analysis_dir / "attention_rankings.csv"
    importance_rankings_path = analysis_dir / "importance_rankings.csv"
    ranking_similarity_records_path = analysis_dir / "ranking_similarity_records.csv"
    ranking_similarity_summary_path = analysis_dir / "ranking_similarity_summary.csv"
    ranking_similarity_markdown_path = analysis_dir / "ranking_similarity_summary.md"
    deletion_records_path = analysis_dir / "deletion_records.csv"
    deletion_summary_path = analysis_dir / "deletion_summary.csv"
    deletion_summary_markdown_path = analysis_dir / "deletion_summary.md"
    qualitative_cases_path = analysis_dir / "qualitative_cases.md"
    confidence_curve_path = analysis_dir / "confidence_drop_curve.html"
    analysis_metadata_path = analysis_dir / "analysis_metadata.json"

    selection_df.to_csv(sampled_examples_path, index=False)
    attention_df.to_csv(attention_rankings_path, index=False)
    importance_df.to_csv(importance_rankings_path, index=False)
    similarity_df.to_csv(ranking_similarity_records_path, index=False)
    similarity_summary_df.to_csv(ranking_similarity_summary_path, index=False)
    deletion_df.to_csv(deletion_records_path, index=False)
    summary_df.to_csv(deletion_summary_path, index=False)

    similarity_markdown = _dataframe_to_markdown(
        similarity_summary_df,
        float_columns=[
            "mean_ranked_token_count",
            "mean_spearman_rank_correlation",
            "std_spearman_rank_correlation",
        ],
    )
    write_text(ranking_similarity_markdown_path, similarity_markdown + "\n")

    summary_markdown = _dataframe_to_markdown(
        summary_df,
        float_columns=[
            "k_setting_value",
            "mean_confidence_drop",
            "std_confidence_drop",
            "mean_gold_label_prob_drop",
            "mean_target_prob_drop",
            "flip_rate",
            "becomes_incorrect_rate",
            "mean_delta_prob_negative",
            "mean_delta_prob_positive",
        ],
    )
    write_text(deletion_summary_markdown_path, summary_markdown + "\n")

    case_markdown_lines = [
        "# Qualitative Importance Cases",
        "",
        "Attention importance is defined as the last-layer mean attention from `[CLS]` to each non-special token.",
        "Gradient × input is computed against the original predicted-class logit and ranked by absolute attribution magnitude.",
        "Leave-one-out importance is defined as the drop in original predicted-class probability after deleting one token at a time.",
        "Tokens remain in wordpiece form for this DistilBERT SST-2 implementation.",
        "",
    ]
    if case_rows:
        case_df = pd.DataFrame(case_rows)
        case_df["max_confidence_drop"] = case_df[
            [f"{method}_confidence_drop" for method in ranking_methods]
        ].max(axis=1)
        case_df = case_df.sort_values("max_confidence_drop", ascending=False).head(config.case_study_count)
        for _, row in case_df.iterrows():
            case_markdown_lines.extend(
                [
                    f"## Validation Example {int(row['validation_index'])}",
                    "",
                    f"- Sentence: {row['sentence']}",
                    f"- Gold label: {row['label_name']}",
                    f"- Base prediction: {row['base_prediction_name']}",
                    f"- Base probabilities: negative={row['base_prob_negative']:.4f}, positive={row['base_prob_positive']:.4f}",
                    f"- Case deletion setting: {row['case_setting_type']}={row['case_setting_value']} (actual_k={int(row['case_actual_k'])})",
                    "",
                ]
            )
            for method_name in ranking_methods:
                case_markdown_lines.extend(
                    [
                        f"### {method_name}",
                        "",
                        f"- Top tokens: {row[f'{method_name}_top_tokens']}",
                        f"- Deleted tokens: {row[f'{method_name}_deleted_tokens']}",
                        f"- New probabilities after deletion: negative={row[f'{method_name}_new_prob_negative']:.4f}, positive={row[f'{method_name}_new_prob_positive']:.4f}",
                        f"- Confidence drop: {row[f'{method_name}_confidence_drop']:.4f}",
                        f"- Target probability drop: {row[f'{method_name}_target_prob_drop']:.4f}",
                        f"- Prediction flipped: {bool(row[f'{method_name}_prediction_flip'])}",
                        "",
                    ]
                )
    else:
        case_markdown_lines.append("No qualitative case rows were generated for the current deletion settings.")
        case_markdown_lines.append("")
    write_text(qualitative_cases_path, "\n".join(case_markdown_lines))

    figure = px.line(
        summary_df.copy(),
        x="actual_k",
        y="mean_confidence_drop",
        color="method",
        markers=True,
        title="Confidence Drop After Token Deletion",
        labels={
            "actual_k": "Deleted token count",
            "mean_confidence_drop": "Mean confidence drop",
            "method": "Deletion method",
        },
    )
    figure.write_html(str(confidence_curve_path))

    write_json(
        analysis_metadata_path,
        {
            "sample_size": int(len(selection_df)),
            "analysis_subset": analysis_subset,
            "analyze_only_correct_predictions": analyze_only_correct_predictions,
            "seed": seed,
            "top_k_values": top_k_values,
            "top_ratios": top_ratios,
            "random_trials": config.random_trials,
            "ranking_methods": ranking_methods,
            "importance_definitions": {
                "attention": "Mean attention over heads in the last layer, using the [CLS] query to score non-special tokens.",
                "grad_x_input": "Absolute gradient × input attribution magnitude against the original predicted-class logit, aggregated across embedding dimensions.",
                "loo": "Drop in original predicted-class probability after deleting one token at a time.",
            },
            "ranking_similarity_metric": "Spearman rank correlation over shared non-special token positions.",
            "token_granularity": "wordpiece",
        },
    )
    _announce(
        "Analysis",
        (
            f"Completed analysis. similarity_summary={ranking_similarity_summary_path}, "
            f"deletion_summary={deletion_summary_path}, qualitative_cases={qualitative_cases_path}"
        ),
    )

    return AnalysisResult(
        checkpoint_dir=Path(config.checkpoint_dir),
        analysis_dir=analysis_dir,
        sampled_examples_path=sampled_examples_path,
        attention_rankings_path=attention_rankings_path,
        importance_rankings_path=importance_rankings_path,
        ranking_similarity_records_path=ranking_similarity_records_path,
        ranking_similarity_summary_path=ranking_similarity_summary_path,
        ranking_similarity_markdown_path=ranking_similarity_markdown_path,
        deletion_records_path=deletion_records_path,
        deletion_summary_path=deletion_summary_path,
        deletion_summary_markdown_path=deletion_summary_markdown_path,
        qualitative_cases_path=qualitative_cases_path,
        confidence_curve_path=confidence_curve_path,
        analysis_metadata_path=analysis_metadata_path,
    )
