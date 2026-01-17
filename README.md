# PMIScore

PMIScore is a framework for evaluating dialogue response quality using Pointwise Mutual Information (PMI). This repository contains the supplementary material for the paper "PMIScore: PMI-Based Scoring for Dialogue Response Quality Assessment".

## Overview

The PMIScore framework provides a comprehensive pipeline for:
- **Data Processing**: Generate synthetic datasets with controlled PMI distributions and process empirical dialogue data (DSTC11)
- **Embedding Generation**: Generate embeddings using 6 state-of-the-art LLMs via vLLM
- **Training & Evaluation**: Train neural network and KDE-based scoring heads with 5-round negative sampling
- **Visualization**: Generate publication-ready figures and LaTeX tables for research papers

## Project Structure

```
PMIScore/
├── config.py                    # Shared configuration module
├── data_processing.py           # Module 1: Dataset generation & preprocessing
├── embedding_generation.py      # Module 2: Embedding generation & MEEP scoring
├── train_and_evaluate.py        # Module 3: Training & evaluation of scoring heads
├── aggregate_results.py          # Module 3.5: Results aggregation & metric computation
├── visualize_and_report.py      # Module 4: Visualization & LaTeX table generation
├── pyproject.toml               # Project dependencies (managed with uv)
├── uv.lock                      # Locked dependency versions
├── datasets/                    # Output from Module 1 (gitignored)
│   ├── synthetic/
│   │   ├── diagonal/
│   │   ├── independent/
│   │   └── block/
│   └── empirical/
│       ├── en/
│       └── zh/
├── embeddings/                  # Output from Module 2 (gitignored)
│   ├── synthetic/
│   └── empirical/
├── results/                     # Output from Module 3 (gitignored)
│   ├── synthetic/
│   └── empirical/
├── meep_scores/                 # Output from Module 2 (gitignored)
│   ├── synthetic/
│   └── empirical/
└── analysis_report/
    ├── all_results_raw.csv
    ├── aggregated_results.csv
    ├── table_empirical.csv
    ├── table_synthetic.csv
    ├── tables_generated.tex
    ├── Figure1_Synthetic_Pearson_MSE.{png,pdf}
    ├── Figure2_Empirical_AUC_Spearman.{png,pdf}
    └── Figure3_Regression_Scatter_{Dataset}.{png,pdf}
```

## Installation

This project uses `uv` for dependency management.

### Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for Modules 2 and 3)
- At least 32GB RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/nbdhhzh/PMIScore.git
cd PMIScore

# Install dependencies using uv
pip install uv
uv sync
```

## Quick Start

The PMIScore pipeline consists of 5 modules that must be run sequentially:

```bash
# Module 1: Generate and process datasets
uv run data_processing.py

# Module 2: Generate embeddings (GPU required)
uv run embedding_generation.py

# Module 3: Train and evaluate scoring heads (GPU recommended)
uv run train_and_evaluate.py

# Module 3.5: Aggregate results and compute metrics
uv run aggregate_results.py

# Module 4: Generate figures and LaTeX tables
uv run visualize_and_report.py
```

**Note**: After running `uv sync`, use `uv run` instead of `python` to ensure the correct virtual environment is activated. The code respects existing data files and will skip regeneration if files already exist.

## Module Details

### Module 1: Dataset Processing (`data_processing.py`)

**Purpose**: Generate synthetic datasets and download/process empirical datasets (DSTC11).

**Synthetic Datasets**: Three modes with controlled PMI distributions:
- **Diagonal**: High PMI along diagonal
- **Independent**: Zero PMI (independent context-response)
- **Block**: High PMI within blocks

**Empirical Datasets**: DSTC11 Track 4 dialogue data
- English dialogues (`en`)
- Chinese dialogues (`zh`)

**Usage**:
```bash
uv run data_processing.py
```

**Optional API Paraphrasing**: 
Set environment variable to enable paraphrasing:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Output**:
```
datasets/synthetic/{diagonal,independent,block}/
  - train.csv, val.csv, test.csv
  - train_with_negatives.csv, val_with_negatives.csv, test_with_negatives.csv

datasets/empirical/{en,zh}/
  - train.csv, val.csv, test.csv
  - train_with_negatives.csv, val_with_negatives.csv, test_with_negatives.csv
  - human.csv
```

**Key Parameters** (in `config.py`):
- `SYN_SIZE = 20`: Vocabulary size for synthetic data
- `SYN_SAMPLES = 5000`: Number of samples per synthetic dataset
- `EMP_SAMPLES = 5000`: Number of samples per empirical dataset
- `TRAIN_NEG_RATIO = 15`: Number of negative samples per positive

---

### Module 2: Embedding Generation (`embedding_generation.py`)

**Purpose**: Generate embeddings for 6 models and compute MEEP direct scores.

**Supported Models**:
1. `context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16`
2. `microsoft/Phi-4-mini-instruct`
3. `Qwen/Qwen3-0.6B`
4. `Qwen/Qwen3-1.7B`
5. `Qwen/Qwen3-4B`
6. `Qwen/Qwen3-8B`

**Embedding Types**:
- `context_embeddings`: Raw context embeddings
- `response_embeddings`: Raw response embeddings
- `prompt_embeddings`: Joint context-response embeddings (MEEP prompt)
- `prompt_pmi_embeddings`: Joint embeddings (PMI prompt)
- `meta.csv`: Metadata with PMI values and labels

**MEEP Direct Scoring**: Generates N=5 text samples for each prompt to compute engagement scores.

**Usage**:
```bash
uv run embedding_generation.py
```

**Requirements**:
- GPU with sufficient VRAM (depends on model size)
- vLLM installed

**Output**:
```
embeddings/{synthetic,empirical}/{dataset}/{model}/
  - train_context_embeddings.npy
  - train_response_embeddings.npy
  - train_prompt_embeddings.npy
  - train_meta.csv
  - (and same for val/test)

meep_scores/{synthetic,empirical}/{dataset}/{model}/
  - train_scores.csv (mean, std, and raw scores)
  - (and same for val/test)
```

Note: All `.npy`, `.pkl`, and `.pt` files are gitignored to keep the repository lightweight.

**Key Parameters** (in `config.py`):
- `NUM_MEEP_SAMPLES = 5`: Number of MEEP samples per prompt
- `MEEP_PROMPT_TEMPLATE`: Template for engagement scoring
- `PMI_PROMPT_TEMPLATE`: Template for PMI-based scoring

---

### Module 3: Training & Evaluation (`train_and_evaluate.py`)

**Purpose**: Train scoring heads (NN and KDE) and evaluate on test/human splits.

**Training Methods**:

**Neural Networks** (3-layer MLP):
- `PMI_Pair`: PMI-NCE loss with pair embeddings
- `MINE_Pair`: MINE loss with pair embeddings
- `InfoNCE_Pair`: InfoNCE loss with pair embeddings
- `PMI_Prompt`: PMI-NCE loss with prompt embeddings
- `MINE_Prompt`: MINE loss with prompt embeddings
- `InfoNCE_Prompt`: InfoNCE loss with prompt embeddings

**KDE Estimators**:
- `KDE_Pair`: KDE on pair difference embeddings
- `KDE_Prompt`: KDE on prompt embeddings

**Training Process**:
1. Load embeddings from Module 2
2. For T=5 rounds, sample different negative samples (3 per positive per round)
3. Train models and perform inference on test/human splits
4. Save models and inference results

**Usage**:
```bash
uv run train_and_evaluate.py
```

**Requirements**:
- GPU recommended (faster training)
- PyTorch

**Output**:
```
results/{synthetic,empirical}/{dataset}/{model}/
  - round_{t}_{name}_model.pt (NN models)
  - round_{t}_{name}_model.pkl (KDE models)
  - round_{t}_test_inference_results.csv (Test scores)
  - round_{t}_human_inference_results.csv (Human scores)
```

Note: All `.npy`, `.pkl`, and `.pt` files are gitignored to keep the repository lightweight.

**Key Parameters** (in `config.py`):
- `NUM_ROUNDS = 5`: Number of training rounds
- `NEG_SAMPLES_USED = 3`: Negatives used per round
- `EPOCHS = 100`: Training epochs
- `BATCH_SIZE = 256`: Batch size
- `LR = 1e-3`: Learning rate
- `KDE_PCA_DIM = 128`: PCA dimension for KDE

---

### Module 3.5: Results Aggregation (`aggregate_results.py`)

**Purpose**: Aggregate inference results from multiple training rounds and compute comprehensive metrics.

**Key Functions**:
- Reads scattered inference results from `train_and_evaluate.py` (round-specific CSV files)
- Loads Direct MEEP scores and reconstructs sampling indices
- Computes metrics for both Synthetic and Empirical datasets
- Generates consolidated CSV files required for visualization

**Metrics Computed**:

**Synthetic Datasets**:
- AUC: Binary classification performance (positive vs negative)
- Spearman_PMI: Correlation between predicted scores and ground-truth PMI
- Pearson_PMI: Linear correlation with ground-truth PMI
- MSE_PMI: Mean squared error against ground-truth PMI

**Empirical Datasets**:
- AUC: Binary classification performance on test set
- Spearman_Engaging: Correlation with human engagement scores
- Pearson_Engaging: Linear correlation with human engagement scores
- Spearman_Relevant: Correlation with human relevance scores
- Pearson_Relevant: Linear correlation with human relevance scores

**Usage**:
```bash
uv run aggregate_results.py
```

**Output**:
```
analysis_report/
  - all_results_raw.csv          # Raw round-level metrics (required by Module 4)
  - aggregated_results.csv        # Pooled statistics across models and rounds
  - table_synthetic.csv          # Synthetic dataset performance table
  - table_empirical.csv          # Empirical dataset performance table
```

**Key Features**:
- Handles both Synthetic and Empirical datasets
- Processes Direct MEEP scores with proper round-based sampling
- Computes pooled statistics (mean ± std) across 5 training rounds
- Generates display tables for paper figures

**Key Parameters** (in `config.py`):
- `NUM_ROUNDS = 5`: Number of training rounds
- `NEG_SAMPLES_USED = 3`: Negatives used per round
- `GROUP_SIZE = 16`: Total samples per group (1 pos + 15 neg)
- `NUM_MEEP_SCORES = 5`: Number of Direct MEEP scores per sample

---

### Module 4: Visualization & Reporting (`visualize_and_report.py`)

**Purpose**: Generate publication-ready figures and LaTeX tables.

**Figures Generated**:

1. **Figure 1**: Synthetic Performance
   - Left: Pearson correlation (higher is better)
   - Right: Mean Squared Error (lower is better)

2. **Figure 2**: Empirical Performance
   - Left: AUC Score (higher is better)
   - Right: Spearman correlation (higher is better)

3. **Figure 3**: Regression Scatter Plots
   - Predicted PMI vs Ground-Truth PMI for each method
   - Separate files for each dataset (diagonal, independent, block)

**LaTeX Tables Generated**:

1. **Table 1**: Synthetic Results
   - Columns: Dataset, Model, PMIScore, MINE, InfoNCE, KDE, MEEP
   - Metrics: Pearson ρ, MSE

2. **Table 2**: Empirical Results
   - Columns: Dataset, Model, PMIScore, MINE, InfoNCE, KDE, MEEP
   - Metrics: AUC, Spearman ρ

**Usage**:
```bash
uv run visualize_and_report.py
```

**Prerequisites**:
- Must run `aggregate_results.py` first to generate `all_results_raw.csv`

**Output**:
```
analysis_report/
  - Figure1_Synthetic_Pearson_MSE.{png,pdf}
  - Figure2_Empirical_AUC_Spearman.{png,pdf}
  - Figure3_Regression_Scatter_{Dataset}.{png,pdf}
  - table_empirical.csv
  - table_synthetic.csv
  - tables_generated.tex
  - all_results_raw.csv
  - aggregated_results.csv
```

**Key Features**:
- Best values are bolded in tables
- Error bars show standard error of the mean (SEM)
- Figures saved in both PNG (300 DPI) and PDF formats

## Configuration

All configuration parameters are centralized in `config.py`. Key parameters include:

```python
class Config:
    # Paths
    BASE_DIR = Path("./")
    DATA_DIR = BASE_DIR / "datasets"
    EMBEDDING_DIR = BASE_DIR / "embeddings"
    RESULT_DIR = BASE_DIR / "results"
    OUTPUT_DIR = BASE_DIR / "analysis_report"

    # Dataset sizes
    SYN_SIZE = 20
    SYN_SAMPLES = 5000
    EMP_SAMPLES = 5000

    # Negative sampling
    TRAIN_NEG_RATIO = 15
    NUM_ROUNDS = 5
    NEG_SAMPLES_USED = 3

    # Training
    EPOCHS = 100
    BATCH_SIZE = 256
    LR = 1e-3

    # KDE
    KDE_PCA_DIM = 128

    # Models
    MODELS = [
        "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
    ]
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'vllm'`
- **Solution**: Install vLLM: `pip install vllm`

**Issue**: CUDA out of memory during embedding generation
- **Solution**: Use a smaller model or reduce batch size. Consider using models with fewer parameters (e.g., Qwen3-0.6B).

**Issue**: File not found errors
- **Solution**: Ensure previous modules have been run successfully. The pipeline is sequential: Module 1 → Module 2 → Module 3 → Module 4.

**Issue**: Slow training in Module 3
- **Solution**: Use GPU. If using CPU, reduce `BATCH_SIZE` and `EPOCHS` in `config.py`.

**Issue**: Paraphrasing API errors in Module 1
- **Solution**: Set `OPENROUTER_API_KEY` environment variable or skip paraphrasing (not required).

### Existing Data Files

The code respects existing data files and will skip regeneration if files already exist. To force regeneration:
- Delete the specific output directory
- Rerun the module

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{pmiscore2024,
  title={PMIScore: PMI-Based Scoring for Dialogue Response Quality Assessment},
  author={Your Name et al.},
  journal={Journal/Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

This work uses the following resources:
- DSTC11 Track 4 dataset
- vLLM for efficient LLM inference
- Hugging Face Transformers for model access

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
