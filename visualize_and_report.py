"""
Module 4: Visualization and LaTeX Table Generation

This module generates publication-ready figures and LaTeX tables for the paper.

Outputs:
- Figure 1: Synthetic performance bar charts (Pearson correlation & MSE)
- Figure 2: Empirical performance bar charts (AUC & Spearman correlation)
- Figure 3: Regression scatter plots for each dataset
- LaTeX Table 1: Synthetic results (Pearson ρ, MSE)
- LaTeX Table 2: Empirical results (AUC, Spearman ρ)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Import configuration from config module
import config


# ==========================================
# CONFIGURATION
# ==========================================

class VisualizationConfig:
    """Visualization-specific configuration."""

    # Base directories
    BASE_DIR = config.Config.BASE_DIR
    RAW_FILE = BASE_DIR / "analysis_report" / "all_results_raw.csv"
    RESULTS_DIR = BASE_DIR / "results"
    MEEP_DIR = BASE_DIR / "meep_scores"

    # Output directory for plots
    OUTPUT_DIR = BASE_DIR / "analysis_report"

    # Colors for methods
    COLORS = {
        "PMIScore": "#1f77b4",
        "PMIScore-Pair": "#17becf",
        "MINE": "#ff7f0e",
        "InfoNCE": "#2ca02c",
        "KDE": "#d62728",
        "MEEP": "#9467bd",
        "Direct_MEEP": "#9467bd"
    }

    # Method name mapping (internal -> display)
    METHOD_MAP = {
        "PMI_Prompt": "PMIScore",
        "PMI_Pair": "PMIScore-Pair",
        "MINE_Prompt": "MINE",
        "InfoNCE_Prompt": "InfoNCE",
        "KDE_Prompt": "KDE",
        "Direct_MEEP": "MEEP"
    }

    # Target methods for visualization (figures)
    TARGET_METHODS = ["PMIScore", "PMIScore-Pair", "MINE", "InfoNCE", "KDE", "MEEP"]

    # Methods for LaTeX tables (includes all methods, MEEP shown as "-" for OpenRouter models)
    TABLE_METHODS = ["PMIScore", "PMIScore-Pair", "MINE", "InfoNCE", "KDE", "MEEP"]
    
    # OpenRouter model names (tables only, excluded from figures)
    OPENROUTER_MODEL_NAMES = config.Config.OPENROUTER_MODEL_NAMES


# Create output directory
VisualizationConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set matplotlib font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']


# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================

def load_and_aggregate_pooled():
    """
    Load raw results and compute pooled statistics (mean + SEM).

    Aggregation strategy:
    1. Intra-Model: Average over 5 rounds for each model
    2. Inter-Model: Compute mean and SEM across models
    
    Note: Excludes OpenRouter models (tables only, no figures).

    Returns:
        DataFrame with columns: Dataset_Type, Dataset, Method_Display, metric_mu, metric_sigma
    """
    if not VisualizationConfig.RAW_FILE.exists():
        print(f"Error: {VisualizationConfig.RAW_FILE} not found.")
        print("Please run the analysis script first to generate all_results_raw.csv")
        return None

    df = pd.read_csv(VisualizationConfig.RAW_FILE)
    df['Method_Display'] = df['Method'].map(VisualizationConfig.METHOD_MAP).fillna(df['Method'])
    df = df[df['Method_Display'].isin(VisualizationConfig.TARGET_METHODS)].copy()
    
    # Exclude OpenRouter models from figures (they don't have MEEP scores)
    if 'Model' in df.columns:
        df = df[~df['Model'].isin(VisualizationConfig.OPENROUTER_MODEL_NAMES)].copy()

    # Identify metric columns
    metric_cols = [c for c in df.columns if c not in [
        'Dataset_Type', 'Dataset', 'Model', 'Method', 'Method_Display', 'Round'
    ]]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]

    # Step 1: Intra-Model Aggregation (average over rounds)
    df_model_avg = df.groupby(
        ['Dataset_Type', 'Dataset', 'Method_Display', 'Model']
    )[metric_cols].mean().reset_index()

    # Step 2: Inter-Model Statistics (mean + SEM across models)
    final_agg = df_model_avg.groupby(
        ['Dataset_Type', 'Dataset', 'Method_Display']
    )[metric_cols].agg(['mean', 'sem']).reset_index()

    # Step 3: Flatten columns
    result_df = final_agg[['Dataset_Type', 'Dataset', 'Method_Display']].copy()

    for col in metric_cols:
        result_df[f"{col}_mu"] = final_agg[col]['mean']
        result_df[f"{col}_sigma"] = final_agg[col]['sem']

    return result_df


def get_raw_regression_data(dataset_name, n_samples=2000):
    """
    Load raw regression data for scatter plots.

    Args:
        dataset_name: Name of the dataset (e.g., 'block', 'diagonal')
        n_samples: Maximum number of samples to use

    Returns:
        DataFrame with columns: true_pmi, PMIScore, MINE, InfoNCE, KDE, MEEP
    """
    syn_dir = VisualizationConfig.RESULTS_DIR / "synthetic" / dataset_name
    if not syn_dir.exists():
        return None

    model_dirs = [d for d in syn_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        return None

    # Prefer Qwen3-4B model
    target_model = model_dirs[0]
    for d in model_dirs:
        if "Qwen3-4B" in d.name:
            target_model = d
            break

    print(f"  -> Loading raw data from {target_model.name}...")

    res_file = target_model / "round_1_test_inference_results.csv"
    if not res_file.exists():
        return None

    df = pd.read_csv(res_file)

    # Keep only positive samples
    if 'is_positive' in df.columns:
        df = df[df['is_positive'] == 1].copy()
    if df.empty:
        return None

    # Calculate true PMI
    eps = 1e-12
    p_xy = df['p_xy'].clip(lower=eps)
    p_x = df['p_x'].clip(lower=eps)
    p_y = df['p_y'].clip(lower=eps)
    df['true_pmi'] = np.log(p_xy) - np.log(p_x) - np.log(p_y)

    # Load MEEP scores
    meep_file = VisualizationConfig.MEEP_DIR / "synthetic" / dataset_name / target_model.name / "test_scores.csv"
    if meep_file.exists():
        try:
            m_df = pd.read_csv(meep_file)
            min_len = min(len(pd.read_csv(res_file)), len(m_df))
            m_raw = m_df.iloc[:min_len].loc[df.index]

            def parse_mean(x):
                try:
                    vals = json.loads(x)
                    valid = [v for v in vals if v is not None]
                    return np.mean(valid) if valid else np.nan
                except:
                    return np.nan

            df['MEEP'] = m_raw['scores_raw'].apply(parse_mean)
        except:
            pass

    # Build final data frame
    final_data = pd.DataFrame({'true_pmi': df['true_pmi']})
    rev_map = {v: k for k, v in VisualizationConfig.METHOD_MAP.items()}

    for disp_name in VisualizationConfig.TARGET_METHODS:
        raw_name = rev_map.get(disp_name, disp_name)
        for cand in [f"score_{raw_name}", raw_name, disp_name, f"score_{disp_name}"]:
            if cand in df.columns:
                final_data[disp_name] = df[cand]
                break
            if disp_name == "MEEP" and "MEEP" in df.columns:
                final_data["MEEP"] = df["MEEP"]
                break

    if len(final_data) > n_samples:
        final_data = final_data.sample(n=n_samples, random_state=42)

    return final_data.dropna()


def z_normalize(x):
    """Z-score normalization."""
    if len(x) < 2:
        return x
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


# ==========================================
# LATEX TABLE GENERATION
# ==========================================

def format_latex_cell(mean, std, is_best):
    """
    Format a single cell for LaTeX table.

    Args:
        mean: Mean value
        std: Standard deviation
        is_best: Whether this is the best value (will be bolded)

    Returns:
        LaTeX formatted string
    """
    val_str = f"{mean:.3f}"
    std_str = f"({std:.3f})"

    if is_best:
        content = "\\makecell{\\textbf{" + val_str + "} \\\\ {\\scriptsize " + std_str + "}}"
    else:
        content = "\\makecell{" + val_str + " \\\\ {\\scriptsize " + std_str + "}}"

    return content

def print_latex_tables():
    """
    Generate LaTeX tables for Synthetic and Empirical results and save to file.
    
    Outputs: tables_generated.tex in the output directory.
    """
    if not VisualizationConfig.RAW_FILE.exists():
        print("Raw file not found, skipping tables.")
        return

    output_path = VisualizationConfig.OUTPUT_DIR / "tables_generated.tex"
    print("\n" + "="*50)
    print(f">>> GENERATING LATEX TABLES")
    print(f"    Saving to: {output_path}")
    print("="*50)

    df = pd.read_csv(VisualizationConfig.RAW_FILE)
    df['Method_Display'] = df['Method'].map(VisualizationConfig.METHOD_MAP).fillna(df['Method'])
    df = df[df['Method_Display'].isin(VisualizationConfig.TABLE_METHODS)].copy()

    with open(output_path, "w", encoding="utf-8") as f:
        
        # Helper to mimic print behavior but write to file
        def write_line(text=""):
            f.write(str(text) + "\n")

        write_line("% ==========================================")
        write_line("% AUTOMATICALLY GENERATED LATEX TABLES")
        write_line(f"% Generated from: {VisualizationConfig.RAW_FILE.name}")
        write_line("% ==========================================\n")

        # ==========================
        # Table 1: Synthetic
        # ==========================
        write_line("\n% --- Table 1: Synthetic (Pearson & MSE) ---")
        syn_df = df[df['Dataset_Type'] == 'Synthetic'].copy()

        rho_col = 'Pearson_PMI'
        mse_col = 'MSE_PMI'

        if not syn_df.empty:
            # Aggregate by Dataset, Model, Method
            syn_agg = syn_df.groupby(
                ['Dataset', 'Model', 'Method_Display']
            )[[rho_col, mse_col]].agg(['mean', 'std']).reset_index()

            datasets = sorted(syn_agg['Dataset'].unique())

            # Table header
            write_line(r"\begin{table*}[!ht]")
            write_line(r"\centering")
            write_line(r"\small")
            write_line(r"\setlength{\tabcolsep}{3.5pt}")
            write_line(r"\begin{tabular}{ll" + "cc" * len(VisualizationConfig.TABLE_METHODS) + "}")
            write_line(r"\toprule")

            # Header rows
            header1 = r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} "
            header2 = " & "
            for m in VisualizationConfig.TABLE_METHODS:
                header1 += r"& \multicolumn{2}{c}{" + m + "} "
                header2 += r"& $\rho$ & MSE "
            write_line(header1 + r"\\")
            write_line(header2 + r"\\")
            write_line(r"\midrule")

            # Data rows
            for i, ds in enumerate(datasets):
                ds_df = syn_agg[syn_agg['Dataset'] == ds]
                models = sorted(ds_df['Model'].unique())

                # Dataset multirow
                write_line(r"\multirow{" + str(int(len(models) * 1.5)) + r"}*}{" + ds.capitalize() + r"}")

                for model in models:
                    row_str = f" & {model}"
                    if "Llama" in model:
                        row_str = f" & Llama-3.2-3B-Instruct"

                    m_df = ds_df[ds_df['Model'] == model].set_index('Method_Display')

                    # Find best values for this model
                    valid_rhos = m_df[rho_col]['mean']
                    valid_mses = m_df[mse_col]['mean']
                    best_rho = valid_rhos.max() if not valid_rhos.empty else -999
                    best_mse = valid_mses.min() if not valid_mses.empty else 9999

                    for method in VisualizationConfig.TABLE_METHODS:
                        if method in m_df.index:
                            # Rho
                            r_mu = m_df.loc[method, (rho_col, 'mean')]
                            r_std = m_df.loc[method, (rho_col, 'std')]
                            is_best_r = (r_mu >= best_rho - 1e-6)
                            row_str += " & " + format_latex_cell(r_mu, r_std, is_best_r)

                            # MSE
                            m_mu = m_df.loc[method, (mse_col, 'mean')]
                            m_std = m_df.loc[method, (mse_col, 'std')]
                            is_best_m = (m_mu <= best_mse + 1e-6)
                            row_str += " & " + format_latex_cell(m_mu, m_std, is_best_m)
                        else:
                            row_str += " & - & -"

                    write_line(row_str + r"\\")

                if i < len(datasets) - 1:
                    write_line(r"\midrule")

            write_line(r"\bottomrule")
            write_line(r"\end{tabular}")
            write_line(r"\caption{Performance on Synthetic datasets. Best Pearson $\rho$ (higher) and MSE (lower) per model are \\textbf{bolded}.}")
            write_line(r"\label{tab:synthetic}")
            write_line(r"\end{table*}")

        # ==========================
        # Table 2: Empirical
        # ==========================
        write_line("\n\n% --- Table 2: Empirical (AUC & Spearman) ---")
        emp_df = df[df['Dataset_Type'] == 'Empirical'].copy()

        auc_col = 'AUC'
        sp_col = 'Spearman_Relevant' if 'Spearman_Relevant' in emp_df.columns else 'Spearman'

        if not emp_df.empty:
            emp_agg = emp_df.groupby(
                ['Dataset', 'Model', 'Method_Display']
            )[[auc_col, sp_col]].agg(['mean', 'std']).reset_index()

            datasets_emp = sorted(emp_agg['Dataset'].unique())

            write_line(r"\begin{table*}[!ht]")
            write_line(r"\centering")
            write_line(r"\small")
            write_line(r"\setlength{\tabcolsep}{3.5pt}")
            write_line(r"\begin{tabular}{ll" + "cc" * len(VisualizationConfig.TABLE_METHODS) + "}")
            write_line(r"\toprule")

            header1 = r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} "
            header2 = " & "
            for m in VisualizationConfig.TABLE_METHODS:
                header1 += r"& \multicolumn{2}{c}{" + m + "} "
                header2 += r"& AUC & $\rho$ "
            write_line(header1 + r"\\")
            write_line(header2 + r"\\")
            write_line(r"\midrule")

            for i, ds in enumerate(datasets_emp):
                ds_df = emp_agg[emp_agg['Dataset'] == ds]
                models = sorted(ds_df['Model'].unique())

                write_line(r"\multirow{" + str(int(len(models) * 1.5)) + r"}*}{" + ds.capitalize() + r"}")

                for model in models:
                    row_str = f" & {model}"
                    if "Llama" in model:
                        row_str = f" & Llama-3.2-3B-Instruct"

                    m_df = ds_df[ds_df['Model'] == model].set_index('Method_Display')

                    valid_auc = m_df[auc_col]['mean']
                    valid_sp = m_df[sp_col]['mean']
                    best_auc = valid_auc.max() if not valid_auc.empty else -1
                    best_sp = valid_sp.max() if not valid_sp.empty else -1

                    for method in VisualizationConfig.TABLE_METHODS:
                        if method in m_df.index:
                            # AUC
                            a_mu = m_df.loc[method, (auc_col, 'mean')]
                            a_std = m_df.loc[method, (auc_col, 'std')]
                            is_best_a = (a_mu >= best_auc - 1e-6)
                            row_str += " & " + format_latex_cell(a_mu, a_std, is_best_a)

                            # Spearman
                            s_mu = m_df.loc[method, (sp_col, 'mean')]
                            s_std = m_df.loc[method, (sp_col, 'std')]
                            is_best_s = (s_mu >= best_sp - 1e-6)
                            row_str += " & " + format_latex_cell(s_mu, s_std, is_best_s)
                        else:
                            row_str += " & - & -"

                    write_line(row_str + r"\\")

                if i < len(datasets_emp) - 1:
                    write_line(r"\midrule")

            write_line(r"\bottomrule")
            write_line(r"\end{tabular}")
            write_line(r"\caption{Performance on Empirical datasets. Best AUC (higher) and Spearman $\rho$ (higher) per model are bolded.}")
            write_line(r"\label{tab:empirical}")
            write_line(r"\end{table*}")
        else:
            write_line("% No Empirical data found in raw CSV.")
            
    print("[Done] LaTeX tables saved.")

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def save_and_show(fig, filename):
    """
    Save figure as PNG and PDF, then display.

    Args:
        fig: Matplotlib figure
        filename: Output filename (without extension)
    """
    png_path = VisualizationConfig.OUTPUT_DIR / f"{filename}.png"
    pdf_path = VisualizationConfig.OUTPUT_DIR / f"{filename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"[Saved] {png_path}")
    plt.close(fig)


def plot_figure_1_synthetic(df):
    """
    Generate Figure 1: Synthetic performance bar charts.

    Two subplots:
    - Left: Pearson correlation (higher is better)
    - Right: Mean Squared Error (lower is better)

    Args:
        df: Aggregated DataFrame with metrics
    """
    subset = df[df['Dataset_Type'] == 'Synthetic'].copy()
    datasets = ["diagonal", "block", "independent"]
    methods = VisualizationConfig.TARGET_METHODS
    n_methods = len(methods)
    bar_width = 0.8 / n_methods

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

    # Left: Pearson
    ax = axes[0]
    metric = 'Pearson_PMI'
    for i, method in enumerate(methods):
        m_data = subset[subset['Method_Display'] == method].set_index('Dataset').reindex(datasets)
        ax.bar(
            np.arange(3) - 0.4 + bar_width/2 + i*bar_width,
            m_data[f"{metric}_mu"], bar_width,
            yerr=m_data[f"{metric}_sigma"].fillna(0),
            error_kw={'elinewidth': 1, 'markeredgewidth': 1, 'ecolor': 'black'},
            capsize=2,
            color=VisualizationConfig.COLORS.get(method),
            edgecolor='black',
            alpha=0.9,
            label=method
        )

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([d.title() for d in datasets], fontsize=11)
    ax.set_ylabel('Pearson Correlation (↑)', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.05)

    # Right: MSE
    ax = axes[1]
    metric = 'MSE_PMI'
    for i, method in enumerate(methods):
        m_data = subset[subset['Method_Display'] == method].set_index('Dataset').reindex(datasets)
        ax.bar(
            np.arange(3) - 0.4 + bar_width/2 + i*bar_width,
            m_data[f"{metric}_mu"], bar_width,
            yerr=m_data[f"{metric}_sigma"].fillna(0),
            error_kw={'elinewidth': 1, 'markeredgewidth': 1, 'ecolor': 'black'},
            capsize=2,
            color=VisualizationConfig.COLORS.get(method),
            edgecolor='black',
            alpha=0.9
        )

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([d.title() for d in datasets], fontsize=11)
    ax.set_ylabel('Mean Squared Error (↓)', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 40)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=n_methods,
        frameon=False,
        fontsize=10
    )
    plt.tight_layout()
    save_and_show(fig, 'Figure1_Synthetic_Pearson_MSE')


def plot_figure_2_empirical(df, metric_left='AUC', metric_right='Spearman_Relevant'):
    """
    Generate Figure 2: Empirical performance bar charts.

    Two subplots:
    - Left: AUC Score (higher is better)
    - Right: Spearman correlation (higher is better)

    Args:
        df: Aggregated DataFrame with metrics
        metric_left: Left subplot metric (default: 'AUC')
        metric_right: Right subplot metric (default: 'Spearman_Relevant')
    """
    subset = df[df['Dataset_Type'] == 'Empirical'].copy()
    datasets = sorted(subset['Dataset'].unique())
    methods = VisualizationConfig.TARGET_METHODS
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    ds_labels = ["English", "Chinese"] if len(datasets) == 2 else datasets

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

    # Left: AUC
    ax = axes[0]
    metric = metric_left
    for i, method in enumerate(methods):
        m_data = subset[subset['Method_Display'] == method].set_index('Dataset').reindex(datasets)
        ax.bar(
            np.arange(len(datasets)) - 0.4 + bar_width/2 + i*bar_width,
            m_data[f"{metric}_mu"], bar_width,
            yerr=m_data[f"{metric}_sigma"].fillna(0),
            error_kw={'elinewidth': 1, 'markeredgewidth': 1, 'ecolor': 'black'},
            capsize=2,
            color=VisualizationConfig.COLORS.get(method),
            edgecolor='black',
            alpha=0.9,
            label=method
        )

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(ds_labels, fontsize=11)
    ax.set_ylabel('AUC Score (↑)', fontweight='bold', fontsize=12)
    ax.set_ylim(0.4, 1.0)

    # Right: Spearman
    ax = axes[1]
    metric = metric_right
    for i, method in enumerate(methods):
        m_data = subset[subset['Method_Display'] == method].set_index('Dataset').reindex(datasets)
        ax.bar(
            np.arange(len(datasets)) - 0.4 + bar_width/2 + i*bar_width,
            m_data[f"{metric}_mu"], bar_width,
            yerr=m_data[f"{metric}_sigma"].fillna(0),
            error_kw={'elinewidth': 1, 'markeredgewidth': 1, 'ecolor': 'black'},
            capsize=2,
            color=VisualizationConfig.COLORS.get(method),
            edgecolor='black',
            alpha=0.9
        )

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(ds_labels, fontsize=11)
    ax.set_ylabel('Spearman Correlation (↑)', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=n_methods,
        frameon=False,
        fontsize=10
    )
    plt.tight_layout()
    save_and_show(fig, 'Figure2_Empirical_AUC_Spearman')


def plot_figure_3_regression_scatter(dataset_name):
    """
    Generate Figure 3: Regression scatter plots per dataset.

    Creates a 2x3 subplot showing predicted PMI vs ground-truth PMI
    for each method on a specific dataset.

    Args:
        dataset_name: Name of the dataset to plot (e.g., 'block')
    """
    print(f"Generating scatter plots for {dataset_name}...")
    df = get_raw_regression_data(dataset_name)

    if df is None or df.empty:
        print(f"Skipping {dataset_name} (No data).")
        return

    methods = VisualizationConfig.TARGET_METHODS
    n_methods = len(methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    true_vals = df['true_pmi'].values
    lims = [-4.5, 4.5]

    for i, method in enumerate(methods):
        ax = axes[i]

        if method not in df.columns:
            ax.text(0.5, 0.5, "N/A", ha='center')
            ax.axis('off')
            continue

        yp = df[method].values
        mask = np.isfinite(true_vals) & np.isfinite(yp)
        yt, yp = true_vals[mask], yp[mask]

        if len(yt) > 1:
            if method == "MEEP":
                yp_plot = z_normalize(yp) * np.std(yt) + np.mean(yt)
                subtitle = "(Scaled)"
            else:
                yp_plot = yp
                subtitle = "(Raw)"
            mse_val = mean_squared_error(yt, yp_plot)
        else:
            yp_plot = yp
            mse_val = np.nan
            subtitle = ""

        ax.scatter(yt, yp_plot, color=VisualizationConfig.COLORS.get(method), alpha=0.2, s=50, edgecolor='none')
        ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1)

        ax.set_title(f"{method}\nMSE={mse_val:.3f} {subtitle}", fontweight='bold', fontsize=11)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])

        ax.set_xlabel('Ground-Truth PMI', fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel(f'{dataset_name.capitalize()}\nEstimated PMI', fontweight='bold', fontsize=11)

    for j in range(n_methods, len(axes)):
        axes[j].axis('off')

    save_and_show(fig, f'Figure3_Regression_Scatter_{dataset_name.capitalize()}')


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main function to generate all figures and LaTeX tables."""
    # Step 1: Print LaTeX tables
    print_latex_tables()

    # Step 2: Load data and calculate pooled statistics
    print("\n>>> Step 1: Loading Data & Calculating Pooled Stats (Mean + SEM)...")
    df_pooled = load_and_aggregate_pooled()

    if df_pooled is not None:
        # Step 3: Generate Figure 1 (Synthetic)
        print("\n>>> Step 2: Generating Figure 1 (Synthetic)...")
        plot_figure_1_synthetic(df_pooled)

        # Step 4: Generate Figure 2 (Empirical)
        print("\n>>> Step 3: Generating Figure 2 (Empirical)...")
        plot_figure_2_empirical(df_pooled)

    # Step 5: Generate Figure 3 (Regression Scatters)
    print("\n>>> Step 4: Generating Figure 3 (Regression Scatters)...")
    for dataset_name in ["diagonal", "independent", "block"]:
        plot_figure_3_regression_scatter(dataset_name)

    print("\n[Done] All plots generated.")
    print(f"Output directory: {VisualizationConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
