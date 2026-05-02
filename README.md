# COMP5329 Assignment 2 Project

## Project Focus

This repository studies a single interpretability question:

**Do attention weights reflect token importance in a Transformer classifier?**

The current project uses `distilbert-base-uncased` fine-tuned on the GLUE SST-2 sentiment classification task. The analysis compares three token-ranking signals:

- last-layer `[CLS]` attention
- `gradient × input`
- `leave-one-out (LOO)` deletion

The repository is organized around two notebook entry points:

- [experiment.ipynb](</D:/5329_assignment2/experiment.ipynb>): orchestrates training, evaluation, and token-importance analysis
- [paper_figures.ipynb](</D:/5329_assignment2/paper_figures.ipynb>): orchestrates figure generation for the paper from saved experiment artifacts

## Repository Layout

```text
.
├── AssignmentDetail/           # Assignment brief and reference material
├── experiment.ipynb            # Main experiment console
├── paper/                      # Abstracts, outline, and paper template assets
├── paper_figures.ipynb         # Paper figure console
├── requirements.txt            # Python dependencies
├── results/                    # Checkpoints, metrics, and analysis artifacts
└── src/
    ├── model/                  # Training, evaluation, and analysis pipeline
    └── plotting.py             # Figure-loading and plotting utilities for the paper
```

Important files inside `paper/`:

- [paper/abstract.md](</D:/5329_assignment2/paper/abstract.md>)
- [paper/abstract_zh.md](</D:/5329_assignment2/paper/abstract_zh.md>)
- [paper/paper_outline.md](</D:/5329_assignment2/paper/paper_outline.md>)
- [paper/Template/main.tex](</D:/5329_assignment2/paper/Template/main.tex>)

Important files inside `src/model/`:

- [src/model/api.py](</D:/5329_assignment2/src/model/api.py>): public training, evaluation, and analysis functions
- [src/model/config.py](</D:/5329_assignment2/src/model/config.py>): experiment configuration dataclasses
- [src/model/attention.py](</D:/5329_assignment2/src/model/attention.py>): token-importance ranking and deletion logic
- [src/model/data.py](</D:/5329_assignment2/src/model/data.py>): SST-2 loading and tokenization helpers

## Environment Setup

The notebooks assume a local virtual environment and a project-root working directory.

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use Jupyter from outside the virtual environment, the notebooks may fail to import project dependencies even though the repository itself is correct.

## Experiment Workflow

Open [experiment.ipynb](</D:/5329_assignment2/experiment.ipynb>) and run it from top to bottom.

The notebook is intentionally a thin console. It does not reimplement the modeling logic; instead it builds configs and calls the Python API in `src/model/`.

The workflow is:

1. Define shared settings and the seed list.
2. Build one train/eval/analysis config per seed.
3. Fine-tune DistilBERT on SST-2.
4. Evaluate the saved checkpoint on the validation split.
5. Run token-importance analysis and deletion experiments.
6. Load saved artifacts back into tables and plots for inspection.

## Current Result Structure

The `results/` directory currently contains three artifact groups:

- `results/checkpoints/`: trained model checkpoints for each seed
- `results/metrics/`: run configs, training logs, validation metrics, and validation predictions
- `results/analysis/`: token rankings, similarity summaries, deletion summaries, qualitative cases, and analysis metadata

Examples of analysis artifacts:

- `attention_rankings.csv`
- `importance_rankings.csv`
- `ranking_similarity_summary.csv`
- `deletion_summary.csv`
- `qualitative_cases.md`
- `analysis_metadata.json`

Examples of evaluation artifacts:

- `validation_metrics.json`
- `validation_predictions.csv`
- `run_config.json`

## Figure Generation Workflow

Open [paper_figures.ipynb](</D:/5329_assignment2/paper_figures.ipynb>) after experiment artifacts already exist.

That notebook loads figure functions from [src/plotting.py](</D:/5329_assignment2/src/plotting.py>) and generates the paper-oriented charts, including:

- experiment workflow overview
- validation metric overview across seeds
- ranking similarity summary
- deletion confidence-drop curves
- deletion flip-rate curves
- qualitative token-ranking visualization for a selected case

By default, the notebook writes figure files to `paper/figures/` when you execute the export cell.

## Project Status

The current repository state reflects a notebook-driven workflow with multi-seed experiment outputs already generated. The paper-writing assets now live under `paper/`, while experiment logic and plotting logic are separated into `src/model/` and `src/plotting.py`.

The current analysis configuration uses:

- four seeds: `42`, `43`, `44`, `45`
- SST-2 validation split
- token ranking methods: `attention`, `grad_x_input`, `loo`
- deletion baselines including `random`
- a validation-wide analysis run rather than a small sampled subset

## Notes

- `results/checkpoints/` contains large binary files.
- `results/metrics/` and `results/analysis/` are the primary directories used for paper writing.
- `paper/paper_outline.md` is the planning document for drafting the final report.
