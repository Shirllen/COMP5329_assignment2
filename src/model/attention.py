from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class RankedTokens:
    positions: List[int]
    tokens: List[str]
    scores: List[float]


def _get_candidate_tokens(tokenizer, encoded_batch: Dict[str, torch.Tensor]) -> List[Tuple[int, str]]:
    input_ids = encoded_batch["input_ids"][0].tolist()
    attention_mask = encoded_batch["attention_mask"][0].tolist()
    special_mask = encoded_batch["special_tokens_mask"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return [
        (position, token)
        for position, (token, mask_value, is_special) in enumerate(zip(tokens, attention_mask, special_mask))
        if mask_value == 1 and is_special == 0
    ]


def _build_ranked_tokens(candidates: Sequence[Tuple[int, str]], scores_by_position: Dict[int, float]) -> RankedTokens:
    scored = [
        (position, token, float(scores_by_position[position]))
        for position, token in candidates
        if position in scores_by_position
    ]
    scored.sort(key=lambda item: item[2], reverse=True)
    return RankedTokens(
        positions=[item[0] for item in scored],
        tokens=[item[1] for item in scored],
        scores=[item[2] for item in scored],
    )


def run_batched_inference(model, batch: Dict[str, torch.Tensor], device: torch.device, output_attentions: bool = False):
    model.eval()
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=output_attentions)
    logits = outputs.logits.detach().cpu()
    probabilities = torch.softmax(logits, dim=-1).detach().cpu()
    result = {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": probabilities.argmax(dim=-1),
        "confidence": probabilities.max(dim=-1).values,
    }
    if output_attentions:
        result["attentions"] = outputs.attentions
    return result


def run_inference(model, batch: Dict[str, torch.Tensor], device: torch.device, output_attentions: bool = False):
    batch_result = run_batched_inference(model, batch, device=device, output_attentions=output_attentions)
    prediction = int(batch_result["predictions"][0].item())
    confidence = float(batch_result["confidence"][0].item())
    result = {
        "logits": batch_result["logits"][0:1],
        "probabilities": batch_result["probabilities"][0:1],
        "prediction": prediction,
        "confidence": confidence,
    }
    if output_attentions:
        result["attentions"] = batch_result["attentions"]
    return result


def rank_tokens_by_attention(tokenizer, encoded_batch: Dict[str, torch.Tensor], attentions) -> RankedTokens:
    candidates = _get_candidate_tokens(tokenizer, encoded_batch)
    last_layer = attentions[-1][0].detach().cpu()
    mean_attention = last_layer.mean(dim=0)
    cls_attention = mean_attention[0]
    scores_by_position = {
        position: float(cls_attention[position].item())
        for position, _ in candidates
    }
    return _build_ranked_tokens(candidates, scores_by_position)


def rank_tokens_by_gradient_x_input(
    model,
    tokenizer,
    encoded_batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_label: int,
) -> RankedTokens:
    candidates = _get_candidate_tokens(tokenizer, encoded_batch)
    input_ids = encoded_batch["input_ids"].to(device)
    attention_mask = encoded_batch["attention_mask"].to(device)
    input_embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
    target_logit = outputs.logits[0, target_label]
    gradients = torch.autograd.grad(target_logit, input_embeddings)[0][0].detach().cpu()
    embeddings = input_embeddings[0].detach().cpu()
    scores = (gradients * embeddings).sum(dim=-1).numpy()
    scores_by_position = {
        position: float(scores[position])
        for position, _ in candidates
    }
    return _build_ranked_tokens(candidates, scores_by_position)


def build_deletion_batch(
    encoded_batch: Dict[str, torch.Tensor],
    positions_to_delete: Sequence[int],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    positions = set(positions_to_delete)
    original_input_ids = encoded_batch["input_ids"][0].tolist()
    original_attention_mask = encoded_batch["attention_mask"][0].tolist()
    active_length = int(sum(original_attention_mask))

    active_tokens = original_input_ids[:active_length]
    remaining_tokens = [token_id for idx, token_id in enumerate(active_tokens) if idx not in positions]
    if not remaining_tokens:
        raise ValueError("Deletion removed every token, which should not happen after filtering special tokens.")

    target_length = len(original_input_ids)
    pad_length = target_length - len(remaining_tokens)
    new_input_ids = remaining_tokens + [pad_token_id] * pad_length
    new_attention_mask = [1] * len(remaining_tokens) + [0] * pad_length

    return {
        "input_ids": torch.tensor([new_input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([new_attention_mask], dtype=torch.long),
    }


def build_deletion_batches(
    encoded_batch: Dict[str, torch.Tensor],
    deletions: Sequence[Sequence[int]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    if not deletions:
        raise ValueError("At least one deletion set is required.")
    deletion_batches = [
        build_deletion_batch(encoded_batch=encoded_batch, positions_to_delete=positions, pad_token_id=pad_token_id)
        for positions in deletions
    ]
    return {
        "input_ids": torch.cat([batch["input_ids"] for batch in deletion_batches], dim=0),
        "attention_mask": torch.cat([batch["attention_mask"] for batch in deletion_batches], dim=0),
    }


def rank_tokens_by_leave_one_out(
    model,
    tokenizer,
    encoded_batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_label: int,
    base_target_probability: float,
    pad_token_id: int,
) -> RankedTokens:
    candidates = _get_candidate_tokens(tokenizer, encoded_batch)
    deletion_batch = build_deletion_batches(
        encoded_batch=encoded_batch,
        deletions=[[position] for position, _ in candidates],
        pad_token_id=pad_token_id,
    )
    loo_output = run_batched_inference(model, deletion_batch, device=device, output_attentions=False)
    target_probabilities = loo_output["probabilities"][:, target_label].numpy()
    scores_by_position = {
        position: float(base_target_probability - target_probability)
        for (position, _), target_probability in zip(candidates, target_probabilities)
    }
    return _build_ranked_tokens(candidates, scores_by_position)


def compute_spearman_rank_correlation(
    ranked_tokens_a: RankedTokens,
    ranked_tokens_b: RankedTokens,
) -> float:
    rank_a = {position: rank for rank, position in enumerate(ranked_tokens_a.positions, start=1)}
    rank_b = {position: rank for rank, position in enumerate(ranked_tokens_b.positions, start=1)}
    common_positions = sorted(set(rank_a) & set(rank_b))
    if len(common_positions) < 2:
        return 1.0

    ranks_a = np.asarray([rank_a[position] for position in common_positions], dtype=float)
    ranks_b = np.asarray([rank_b[position] for position in common_positions], dtype=float)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centered_a, centered_b) / denominator)


def summarize_deletion(
    original_probabilities: np.ndarray,
    original_prediction: int,
    new_probabilities: np.ndarray,
    gold_label: int,
) -> Dict[str, float | int]:
    original_probabilities = np.asarray(original_probabilities, dtype=float)
    new_probabilities = np.asarray(new_probabilities, dtype=float)
    original_confidence = float(original_probabilities.max())
    new_confidence = float(new_probabilities.max())
    new_prediction = int(new_probabilities.argmax())

    return {
        "original_prediction": int(original_prediction),
        "new_prediction": new_prediction,
        "original_confidence": original_confidence,
        "new_confidence": new_confidence,
        "confidence_drop": float(original_confidence - new_confidence),
        "delta_prob_negative": float(new_probabilities[0] - original_probabilities[0]),
        "delta_prob_positive": float(new_probabilities[1] - original_probabilities[1]),
        "gold_label_prob_drop": float(original_probabilities[gold_label] - new_probabilities[gold_label]),
        "prediction_flip": int(new_prediction != original_prediction),
        "becomes_incorrect": int(new_prediction != gold_label),
    }


def format_ranked_tokens(tokens: Sequence[str], scores: Sequence[float], top_n: int = 10) -> str:
    pairs = [f"{token} ({score:.4f})" for token, score in list(zip(tokens, scores))[:top_n]]
    return ", ".join(pairs)
