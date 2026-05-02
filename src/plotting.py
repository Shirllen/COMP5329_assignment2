"""Utilities for loading experiment artifacts and generating paper figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_EXPERIMENT_NAME = "distilbert_sst2_multiseed"
METHOD_ORDER = ["attention", "grad_x_input", "loo", "random"]
SIMILARITY_METHOD_ORDER = ["attention", "grad_x_input", "loo"]
METHOD_LABELS = {
    "attention": "Attention",
    "grad_x_input": "Gradient × Input",
    "loo": "Leave-One-Out",
    "random": "Random",
}
METHOD_COLORS = {
    "attention": "#355070",
    "grad_x_input": "#6D597A",
    "loo": "#BC4749",
    "random": "#8D99AE",
}
METRIC_LABELS = {
    "mean_confidence_drop": "Mean confidence drop",
    "flip_rate": "Prediction flip rate",
    "becomes_incorrect_rate": "Becomes incorrect rate",
}
SETTING_LABELS = {
    "top_k": "Deleted token count",
    "ratio": "Deletion ratio",
}


@dataclass(frozen=True)
class ArtifactPaths:
    run_name: str
    checkpoint_dir: Path
    metrics_dir: Path
    analysis_dir: Path
    validation_metrics: Path
    validation_predictions: Path
    ranking_similarity: Path
    deletion_summary: Path
    deletion_records: Path
    importance_rankings: Path
    qualitative_cases: Path


@dataclass
class PaperFigureData:
    project_root: Path
    experiment_name: str
    seeds: list[int]
    inspect_seed: int
    artifact_paths_by_seed: dict[int, ArtifactPaths]
    validation_metrics: pd.DataFrame
    validation_metrics_aggregate: pd.DataFrame
    ranking_similarity: pd.DataFrame
    ranking_similarity_aggregate: pd.DataFrame
    deletion_summary: pd.DataFrame
    deletion_aggregate: pd.DataFrame
    validation_predictions: pd.DataFrame
    importance_rankings: pd.DataFrame
    deletion_records: pd.DataFrame


def find_project_root(start_path: Path | None = None) -> Path:
    start = (start_path or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "model").exists() and (candidate / "experiment.ipynb").exists():
            return candidate
    raise RuntimeError("Could not locate the project root from the current working directory.")


def build_seed_artifact_paths(project_root: Path, experiment_name: str, seed: int) -> ArtifactPaths:
    run_name = f"{experiment_name}_seed{seed}"
    checkpoint_dir = project_root / "results" / "checkpoints" / run_name
    metrics_dir = project_root / "results" / "metrics" / run_name
    analysis_dir = project_root / "results" / "analysis" / run_name
    return ArtifactPaths(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        analysis_dir=analysis_dir,
        validation_metrics=metrics_dir / "validation_metrics.json",
        validation_predictions=metrics_dir / "validation_predictions.csv",
        ranking_similarity=analysis_dir / "ranking_similarity_summary.csv",
        deletion_summary=analysis_dir / "deletion_summary.csv",
        deletion_records=analysis_dir / "deletion_records.csv",
        importance_rankings=analysis_dir / "importance_rankings.csv",
        qualitative_cases=analysis_dir / "qualitative_cases.md",
    )


def build_artifact_index(
    project_root: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    seeds: Sequence[int] = (42, 43, 44, 45),
) -> dict[int, ArtifactPaths]:
    return {
        int(seed): build_seed_artifact_paths(project_root=project_root, experiment_name=experiment_name, seed=int(seed))
        for seed in seeds
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_validation_metrics(artifact_paths_by_seed: dict[int, ArtifactPaths]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, paths in artifact_paths_by_seed.items():
        payload = _load_json(paths.validation_metrics)
        rows.append({"seed": seed, **payload})
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def aggregate_validation_metrics(validation_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_name in ["accuracy", "f1", "loss", "num_examples"]:
        if metric_name not in validation_metrics_df.columns:
            continue
        rows.append(
            {
                "metric": metric_name,
                "mean": float(validation_metrics_df[metric_name].mean()),
                "std": float(validation_metrics_df[metric_name].std(ddof=0)),
                "min": float(validation_metrics_df[metric_name].min()),
                "max": float(validation_metrics_df[metric_name].max()),
            }
        )
    return pd.DataFrame(rows)


def load_ranking_similarity(artifact_paths_by_seed: dict[int, ArtifactPaths]) -> pd.DataFrame:
    frames = []
    for seed, paths in artifact_paths_by_seed.items():
        frame = _load_csv(paths.ranking_similarity).copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["method_a", "method_b", "seed"]).reset_index(drop=True)


def aggregate_ranking_similarity(ranking_similarity_df: pd.DataFrame) -> pd.DataFrame:
    aggregate_df = (
        ranking_similarity_df.groupby(["method_a", "method_b"], dropna=False)
        .agg(
            num_seeds=("seed", "nunique"),
            mean_ranked_token_count_mean=("mean_ranked_token_count", "mean"),
            mean_ranked_token_count_std=("mean_ranked_token_count", "std"),
            mean_spearman_rank_correlation_mean=("mean_spearman_rank_correlation", "mean"),
            mean_spearman_rank_correlation_std=("mean_spearman_rank_correlation", "std"),
        )
        .reset_index()
    )
    for column in aggregate_df.columns:
        if column.endswith("_std"):
            aggregate_df[column] = aggregate_df[column].fillna(0.0)
    return aggregate_df


def load_deletion_summary(artifact_paths_by_seed: dict[int, ArtifactPaths]) -> pd.DataFrame:
    frames = []
    for seed, paths in artifact_paths_by_seed.items():
        frame = _load_csv(paths.deletion_summary).copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["method", "k_setting_type", "k_setting_value", "seed"])
        .reset_index(drop=True)
    )


def aggregate_deletion_summary(deletion_summary_df: pd.DataFrame) -> pd.DataFrame:
    aggregate_df = (
        deletion_summary_df.groupby(["method", "k_setting_type", "k_setting_value", "actual_k"], dropna=False)
        .agg(
            num_seeds=("seed", "nunique"),
            mean_confidence_drop_mean=("mean_confidence_drop", "mean"),
            mean_confidence_drop_std=("mean_confidence_drop", "std"),
            mean_gold_label_prob_drop_mean=("mean_gold_label_prob_drop", "mean"),
            mean_gold_label_prob_drop_std=("mean_gold_label_prob_drop", "std"),
            mean_target_prob_drop_mean=("mean_target_prob_drop", "mean"),
            mean_target_prob_drop_std=("mean_target_prob_drop", "std"),
            flip_rate_mean=("flip_rate", "mean"),
            flip_rate_std=("flip_rate", "std"),
            becomes_incorrect_rate_mean=("becomes_incorrect_rate", "mean"),
            becomes_incorrect_rate_std=("becomes_incorrect_rate", "std"),
        )
        .reset_index()
    )
    for column in aggregate_df.columns:
        if column.endswith("_std"):
            aggregate_df[column] = aggregate_df[column].fillna(0.0)
    return aggregate_df


def load_paper_figure_data(
    project_root: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    seeds: Sequence[int] = (42, 43, 44, 45),
    inspect_seed: int = 42,
) -> PaperFigureData:
    resolved_root = find_project_root(project_root)
    normalized_seeds = [int(seed) for seed in seeds]
    artifact_paths_by_seed = build_artifact_index(
        project_root=resolved_root,
        experiment_name=experiment_name,
        seeds=normalized_seeds,
    )
    validation_metrics = load_validation_metrics(artifact_paths_by_seed)
    ranking_similarity = load_ranking_similarity(artifact_paths_by_seed)
    deletion_summary = load_deletion_summary(artifact_paths_by_seed)
    inspect_paths = artifact_paths_by_seed[int(inspect_seed)]
    return PaperFigureData(
        project_root=resolved_root,
        experiment_name=experiment_name,
        seeds=normalized_seeds,
        inspect_seed=int(inspect_seed),
        artifact_paths_by_seed=artifact_paths_by_seed,
        validation_metrics=validation_metrics,
        validation_metrics_aggregate=aggregate_validation_metrics(validation_metrics),
        ranking_similarity=ranking_similarity,
        ranking_similarity_aggregate=aggregate_ranking_similarity(ranking_similarity),
        deletion_summary=deletion_summary,
        deletion_aggregate=aggregate_deletion_summary(deletion_summary),
        validation_predictions=_load_csv(inspect_paths.validation_predictions),
        importance_rankings=_load_csv(inspect_paths.importance_rankings),
        deletion_records=_load_csv(inspect_paths.deletion_records),
    )


def _apply_figure_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        font={"family": "Helvetica", "size": 13},
        title={"x": 0.02, "xanchor": "left"},
        margin={"l": 60, "r": 40, "t": 90, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def plot_experiment_workflow() -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(visible=False, range=[0, 12.5], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 10], fixedrange=True)

    boxes = [
        (0.4, 6.7, 2.7, 8.7, "Fine-tune<br>DistilBERT<br>(4 seeds)"),
        (3.2, 6.7, 5.5, 8.7, "Run validation<br>predictions"),
        (6.0, 6.7, 9.0, 8.7, "Rank tokens with<br>attention, grad×input,<br>and LOO"),
        (9.5, 6.7, 12.0, 8.7, "Compare rankings<br>with Spearman"),
        (9.5, 3.6, 12.0, 5.6, "Delete top tokens<br>and measure impact"),
        (6.1, 0.8, 8.9, 2.8, "Summarize<br>findings"),
    ]
    for x0, y0, x1, y1, label in boxes:
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line={"color": "#355070", "width": 2},
            fillcolor="#E9F1F7",
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=label,
            showarrow=False,
            font={"size": 13},
            align="center",
        )

    arrows = [
        ((2.7, 7.7), (3.2, 7.7)),
        ((5.5, 7.7), (6.0, 7.7)),
        ((9.0, 7.7), (9.5, 7.7)),
        ((10.75, 6.7), (10.75, 5.6)),
        ((9.5, 3.6), (8.9, 2.8)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#355070",
            text="",
        )

    fig = _apply_figure_style(fig, height=420)
    fig.update_layout(width=1400, margin={"l": 20, "r": 20, "t": 10, "b": 10}, title=None)
    return fig


def plot_validation_metrics_overview(validation_metrics_df: pd.DataFrame) -> go.Figure:
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("loss", "Loss"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in metric_specs], horizontal_spacing=0.08)
    for col_index, (metric_name, metric_label) in enumerate(metric_specs, start=1):
        values = validation_metrics_df[metric_name]
        fig.add_trace(
            go.Bar(
                x=validation_metrics_df["seed"],
                y=values,
                marker_color="#355070",
                text=[f"{value:.4f}" for value in values],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=col_index,
        )
        mean_value = float(values.mean())
        fig.add_hline(
            y=mean_value,
            line_dash="dash",
            line_color="#BC4749",
            annotation_text=f"mean={mean_value:.4f}",
            annotation_position="top left",
            row=1,
            col=col_index,
        )
        fig.update_yaxes(title_text=metric_label, row=1, col=col_index)
        fig.update_xaxes(title_text="Seed", row=1, col=col_index)
    fig.update_layout(title="Validation performance across seeds")
    return _apply_figure_style(fig, height=520)


def plot_ranking_similarity_heatmap(ranking_similarity_aggregate_df: pd.DataFrame) -> go.Figure:
    matrix = pd.DataFrame(
        1.0,
        index=SIMILARITY_METHOD_ORDER,
        columns=SIMILARITY_METHOD_ORDER,
    )
    for row in ranking_similarity_aggregate_df.itertuples(index=False):
        matrix.loc[row.method_a, row.method_b] = row.mean_spearman_rank_correlation_mean
        matrix.loc[row.method_b, row.method_a] = row.mean_spearman_rank_correlation_mean

    display_matrix = matrix.rename(index=METHOD_LABELS, columns=METHOD_LABELS)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=display_matrix.values,
                x=list(display_matrix.columns),
                y=list(display_matrix.index),
                zmin=0.0,
                zmax=1.0,
                colorscale="Blues",
                text=[[f"{value:.3f}" for value in row] for row in display_matrix.values],
                texttemplate="%{text}",
                colorbar={"title": "Mean Spearman"},
            )
        ]
    )
    fig.update_layout(title="Agreement between ranking methods")
    return _apply_figure_style(fig, height=520)


def plot_deletion_metric_curve(
    deletion_aggregate_df: pd.DataFrame,
    metric_name: str = "mean_confidence_drop",
    setting_type: str = "top_k",
) -> go.Figure:
    if metric_name not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric_name: {metric_name}")
    if setting_type not in SETTING_LABELS:
        raise ValueError(f"Unsupported setting_type: {setting_type}")

    mean_column = f"{metric_name}_mean"
    std_column = f"{metric_name}_std"
    filtered = deletion_aggregate_df[deletion_aggregate_df["k_setting_type"] == setting_type].copy()
    x_column = "actual_k" if setting_type == "top_k" else "k_setting_value"
    fig = go.Figure()

    for method in METHOD_ORDER:
        method_frame = filtered[filtered["method"] == method].sort_values(x_column)
        if method_frame.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=method_frame[x_column],
                y=method_frame[mean_column],
                mode="lines+markers",
                name=METHOD_LABELS[method],
                marker={"size": 9},
                line={"width": 3, "color": METHOD_COLORS[method]},
                error_y={"type": "data", "array": method_frame[std_column], "visible": True},
            )
        )

    title_suffix = "fixed top-k deletion" if setting_type == "top_k" else "ratio-based deletion"
    fig.update_layout(title=f"{METRIC_LABELS[metric_name]} under {title_suffix}")
    fig.update_xaxes(title_text=SETTING_LABELS[setting_type])
    fig.update_yaxes(title_text=METRIC_LABELS[metric_name])
    return _apply_figure_style(fig, height=520)


def plot_qualitative_case_rankings(
    importance_rankings_df: pd.DataFrame,
    validation_predictions_df: pd.DataFrame,
    validation_index: int,
    top_n_tokens: int = 8,
) -> go.Figure:
    if validation_predictions_df[validation_predictions_df["validation_index"] == validation_index].empty:
        raise ValueError(f"validation_index={validation_index} was not found in validation_predictions.csv")

    subplot_titles = ["Attention", "Grad \u00d7 Input", "LOO"]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.14,
        column_widths=[0.31, 0.31, 0.38],
    )

    for col_index, method in enumerate(SIMILARITY_METHOD_ORDER, start=1):
        method_frame = (
            importance_rankings_df[
                (importance_rankings_df["validation_index"] == validation_index)
                & (importance_rankings_df["method"] == method)
            ]
            .sort_values("rank")
            .head(top_n_tokens)
            .copy()
        )
        if method_frame.empty:
            continue
        method_frame = method_frame.iloc[::-1]
        method_frame["display_token"] = [
            f"#{int(rank)} {token}"
            for rank, token in zip(method_frame["rank"], method_frame["token"], strict=False)
        ]
        fig.add_trace(
            go.Bar(
                x=method_frame["importance_score"],
                y=method_frame["display_token"],
                orientation="h",
                marker_color=METHOD_COLORS[method],
                showlegend=False,
            ),
            row=1,
            col=col_index,
        )
        fig.update_xaxes(title_text="Score", automargin=True, row=1, col=col_index)
        if col_index == 1:
            fig.update_yaxes(title_text="Token", automargin=True, tickfont={"size": 11}, row=1, col=col_index)
        else:
            fig.update_yaxes(automargin=True, tickfont={"size": 11}, row=1, col=col_index)

    fig.update_annotations(font={"size": 14})
    fig = _apply_figure_style(fig, height=560)
    fig.update_layout(
        title={"text": f"Validation example {validation_index} token rankings", "font": {"size": 18}},
        width=1100,
        margin={"l": 80, "r": 40, "t": 85, "b": 60},
    )
    return fig


def generate_default_paper_figures(
    data: PaperFigureData,
    qualitative_case_index: int,
    top_n_tokens: int = 8,
) -> dict[str, go.Figure]:
    return {
        "fig01_experiment_workflow": plot_experiment_workflow(),
        "fig02_validation_metrics": plot_validation_metrics_overview(data.validation_metrics),
        "fig03_ranking_similarity_heatmap": plot_ranking_similarity_heatmap(data.ranking_similarity_aggregate),
        "fig04_confidence_drop_topk": plot_deletion_metric_curve(
            data.deletion_aggregate,
            metric_name="mean_confidence_drop",
            setting_type="top_k",
        ),
        "fig05_flip_rate_topk": plot_deletion_metric_curve(
            data.deletion_aggregate,
            metric_name="flip_rate",
            setting_type="top_k",
        ),
        "figA1_confidence_drop_ratio": plot_deletion_metric_curve(
            data.deletion_aggregate,
            metric_name="mean_confidence_drop",
            setting_type="ratio",
        ),
        "figA2_flip_rate_ratio": plot_deletion_metric_curve(
            data.deletion_aggregate,
            metric_name="flip_rate",
            setting_type="ratio",
        ),
        f"fig06_case_rankings_seed{data.inspect_seed}_idx{qualitative_case_index}": plot_qualitative_case_rankings(
            data.importance_rankings,
            data.validation_predictions,
            validation_index=qualitative_case_index,
            top_n_tokens=top_n_tokens,
        ),
    }


def save_figures(
    figures: dict[str, go.Figure],
    output_dir: Path,
    *,
    save_png: bool = True,
    save_html: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for figure_name, figure in figures.items():
        html_path = output_dir / f"{figure_name}.html"
        png_path = output_dir / f"{figure_name}.png"
        png_error = ""

        if save_html:
            figure.write_html(str(html_path), include_plotlyjs="cdn")

        png_saved = False
        if save_png:
            try:
                figure.write_image(str(png_path), scale=2)
                png_saved = True
            except Exception as exc:  # pragma: no cover - export availability depends on local setup
                png_error = str(exc)

        rows.append(
            {
                "figure_name": figure_name,
                "html_path": str(html_path) if save_html else "",
                "png_path": str(png_path) if save_png else "",
                "png_saved": png_saved,
                "png_error": png_error,
            }
        )
    return pd.DataFrame(rows)
