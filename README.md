# DistilBERT on SST-2 Experiment Console

This repository keeps the existing DistilBERT-on-SST-2 implementation, but restructures it around one notebook-driven workflow. The notebook is now the single experiment entry point, while the actual training, evaluation, and token-importance analysis logic remains in Python modules.

## Main Entry Point

Open [experiment.ipynb](/Users/gongtianhao/Hub/SydneyUni/Course/COMP5329 Deep Learning/COMP5329_Assignment2/experiment.ipynb).

The notebook is intentionally thin:

- configure `TrainConfig`, `EvalConfig`, and `AnalysisConfig`
- call `train(train_config)`
- call `evaluate(eval_config)`
- call `run_attention_analysis(analysis_config)`
- load saved outputs for ranking-similarity tables, prediction samples, confidence-drop plots, and qualitative cases

To reuse an existing checkpoint, skip the training cell and run only the evaluation or analysis cells.

## Python API

The public API is exported from [src/model/__init__.py](/Users/gongtianhao/Hub/SydneyUni/Course/COMP5329 Deep Learning/COMP5329_Assignment2/src/model/__init__.py):

```python
from src.model import (
    AnalysisConfig,
    EvalConfig,
    TrainConfig,
    evaluate,
    run_attention_analysis,
    train,
)

train_result = train(train_config)
eval_result = evaluate(eval_config)
analysis_result = run_attention_analysis(analysis_config)
```

Configuration boundaries:

- `TrainConfig`: model path, cache path, training hyperparameters, train/eval subset sizes, output directories
- `EvalConfig`: checkpoint path, evaluation split, evaluation batch size, evaluation subset size, metrics directory
- `AnalysisConfig`: checkpoint path, analysis subset, sampling and deletion settings, analysis output directory

## Project Structure

```text
.
├── experiment.ipynb
├── train_baseline.py
├── evaluate_baseline.py
├── attention_analysis.py
├── scripts/
│   ├── run_baseline.sh
│   └── setup_env.sh
└── src/
    └── model/
        ├── __init__.py
        ├── api.py
        ├── attention.py
        ├── common.py
        ├── config.py
        ├── data.py
        └── experiment.py
```

`src/model/experiment.py` remains as a compatibility layer for older imports. The new public interfaces live in `src/model/config.py` and `src/model/api.py`.

## Environment Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## CLI Compatibility

The original script files are still available as thin wrappers around the refactored API:

```bash
python train_baseline.py
python evaluate_baseline.py
python attention_analysis.py
```

They now default to the refactored output layout:

- checkpoints: `results/checkpoints/distilbert_sst2`
- metrics: `results/metrics/distilbert_sst2`
- analysis: `results/analysis/distilbert_sst2`

## Output Files

Training and evaluation:

- `results/checkpoints/distilbert_sst2/`
- `results/metrics/distilbert_sst2/train_metrics.json`
- `results/metrics/distilbert_sst2/run_config.json`
- `results/metrics/distilbert_sst2/validation_metrics.json`
- `results/metrics/distilbert_sst2/validation_predictions.csv`

Token-importance analysis:

- `results/analysis/distilbert_sst2/sampled_validation_examples.csv`
- `results/analysis/distilbert_sst2/attention_rankings.csv`
- `results/analysis/distilbert_sst2/importance_rankings.csv`
- `results/analysis/distilbert_sst2/ranking_similarity_records.csv`
- `results/analysis/distilbert_sst2/ranking_similarity_summary.csv`
- `results/analysis/distilbert_sst2/ranking_similarity_summary.md`
- `results/analysis/distilbert_sst2/deletion_records.csv`
- `results/analysis/distilbert_sst2/deletion_summary.csv`
- `results/analysis/distilbert_sst2/deletion_summary.md`
- `results/analysis/distilbert_sst2/confidence_drop_curve.html`
- `results/analysis/distilbert_sst2/qualitative_cases.md`
- `results/analysis/distilbert_sst2/analysis_metadata.json`

## Notes

- `analysis_subset` is the canonical validation-filter setting. Use `correct_only` or `all`.
- The old CLI flags `--only-correct` and `--include-incorrect` are still accepted in `attention_analysis.py`.
- The analysis stage now computes three deterministic ranking methods on the sampled validation subset: `attention`, `grad_x_input`, and `loo`.
- `loo` is intentionally restricted to the sampled analysis subset because it requires one deletion pass per candidate token.
- Full experiments are not bundled into the notebook output. The notebook reads whatever artifacts your current run produces.
