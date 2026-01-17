# -*- coding: utf-8 -*-
"""
Module: Results Aggregation

This module aggregates inference results from multiple training rounds and datasets
into consolidated CSV files for visualization and reporting.

Key Functions:
- Reads scattered inference results from train_and_evaluate.py
- Loads Direct MEEP scores and reconstructs sampling indices
- Computes comprehensive metrics for Synthetic and Empirical datasets
- Generates all_results_raw.csv (required by visualize_and_report.py)
- Creates display tables for paper figures

Outputs:
- all_results_raw.csv: Raw round-level metrics
- aggregated_results.csv: Pooled statistics across models
- table_synthetic.csv: Synthetic dataset performance table
- table_empirical.csv: Empirical dataset performance table
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, pearsonr
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Import configuration from config module
import config


# ==========================================
# CONFIGURATION
# ==========================================

class AggregationConfig:
    """Configuration for results aggregation."""

    # Paths
    BASE_DIR = config.Config.BASE_DIR
    EMBEDDING_DIR = config.Config.EMBEDDING_DIR
    RESULT_DIR = config.Config.RESULT_DIR
    SCORE_DIR = config.Config.SCORE_DIR
    OUTPUT_DIR = config.Config.OUTPUT_DIR

    # Dataset names
    SYN_DATASETS = config.Config.SYNTHETIC_CASES
    EMP_DATASETS = config.Config.EMPIRICAL_LANGS

    # Replication of sampling config from Module 3
    NUM_ROUNDS = config.Config.NUM_ROUNDS
    NEG_SAMPLES_USED = config.Config.NEG_SAMPLES_USED
    GROUP_SIZE = config.Config.GROUP_SIZE
    NUM_MEEP_SCORES = 5  # Number of Direct MEEP scores per sample


# Create output directory
AggregationConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_true_pmi(df):
    """Calculate true PMI for Synthetic data: log p(xy) - log p(x) - log p(y)"""
    eps = 1e-9
    p_xy = df['p_xy'].clip(lower=eps)
    p_x = df['p_x'].clip(lower=eps)
    p_y = df['p_y'].clip(lower=eps)
    return np.log(p_xy) - np.log(p_x) - np.log(p_y)


def normalize_score(scores, feature_range=(-5, 5)):
    """Normalize scores to specified range using MinMaxScaler."""
    if len(scores) == 0 or np.all(np.isnan(scores)):
        return scores
    non_nan_scores = scores[~np.isnan(scores)].reshape(-1, 1)
    if len(non_nan_scores) == 0:
        return scores
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_non_nan = scaler.fit_transform(non_nan_scores).flatten()

    result = np.full_like(scores, np.nan)
    result[~np.isnan(scores)] = normalized_non_nan
    return result


def z_normalize_scores(scores):
    """Z-normalize scores (mean=0, std=1)."""
    if len(scores) == 0 or np.all(np.isnan(scores)):
        return scores
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    if std == 0:
        return np.zeros_like(scores)
    return (scores - mean) / std


def load_direct_scores(ds_type, ds_name, model_name, split):
    """Load Direct MEEP Scores generated in Module 2."""
    path = AggregationConfig.SCORE_DIR / ds_type / ds_name / model_name / f"{split}_scores.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df['scores_raw'].values
        except Exception:
            return None
    return None


def load_full_meta(ds_type, ds_name, model_name, split):
    """Load the full meta.csv file for a dataset split."""
    base = AggregationConfig.EMBEDDING_DIR / ds_type / ds_name / model_name
    meta_path = base / f"{split}_meta.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)
    return None


def recreate_sampled_indices(meta_df_full, round_num, group_size, neg_samples_used):
    """
    Recreate the indices that were kept during negative sampling.

    From each group of 16 samples (1 positive + 15 negatives), keep:
    - 1 positive sample (always first)
    - 3 negative samples determined by round_num

    This ensures each round uses different negative samples for diversity.

    Args:
        meta_df_full: Full metadata DataFrame
        round_num: Round number (1-5)
        group_size: Size of each group (16)
        neg_samples_used: Number of negatives used per round (3)

    Returns:
        np.array: Indices to keep for this round
    """
    total_samples = len(meta_df_full)
    if total_samples % group_size != 0:
        total_samples = (total_samples // group_size) * group_size

    indices_to_keep = []

    for i in range(0, total_samples, group_size):
        indices_to_keep.append(i)  # Positive sample

        start_relative_neg_idx_in_group = (round_num - 1) * neg_samples_used + 1
        end_relative_neg_idx_in_group = round_num * neg_samples_used + 1

        for j in range(start_relative_neg_idx_in_group, end_relative_neg_idx_in_group):
            if (i + j) < (i + group_size):
                indices_to_keep.append(i + j)

    return np.array(indices_to_keep)


# ==========================================
# PROCESSING FUNCTIONS
# ==========================================

def process_synthetic(ds_name, model_name, model_dir):
    """
    Process Synthetic dataset results for a single model.

    Args:
        ds_name: Dataset name (e.g., 'diagonal', 'independent', 'block')
        model_name: Model name
        model_dir: Path to model results directory

    Returns:
        list: List of metric dictionaries for each round
    """
    metrics_buffer = []

    # Find all round-specific inference files
    result_files = list(model_dir.glob("round_*_test_inference_results.csv"))
    if not result_files:
        return []

    # Load Direct MEEP scores and full metadata
    full_direct_scores_raw = load_direct_scores("synthetic", ds_name, model_name, "test")
    full_test_meta = load_full_meta("synthetic", ds_name, model_name, "test")

    for f_path in result_files:
        round_id_str = f_path.name.split("_")[1]
        round_id = int(round_id_str)
        try:
            df = pd.read_csv(f_path)

            # 1. Prepare Ground Truth
            true_pmi = get_true_pmi(df)
            labels = df['is_positive'].values

            # 2. Process Direct_MEEP: Round 1 -> Index 0, Round 2 -> Index 1 ...
            if full_direct_scores_raw is not None and full_test_meta is not None:
                indices_from_full_data = recreate_sampled_indices(
                    full_test_meta, round_id,
                    AggregationConfig.GROUP_SIZE, AggregationConfig.NEG_SAMPLES_USED
                )

                if len(indices_from_full_data) == len(df):
                    raw_scores_json_for_round = full_direct_scores_raw[indices_from_full_data]

                    # Determine which MEEP score to use (Round 1 -> index 0, etc.)
                    target_idx = (round_id - 1) % AggregationConfig.NUM_MEEP_SCORES

                    current_meep_scores = []
                    for json_str in raw_scores_json_for_round:
                        try:
                            scores_list = json.loads(json_str)
                            # Get the score at the target index
                            if target_idx < len(scores_list):
                                current_meep_scores.append(scores_list[target_idx])
                            else:
                                current_meep_scores.append(np.nan)
                        except (json.JSONDecodeError, TypeError):
                            current_meep_scores.append(np.nan)

                    current_meep_scores = pd.to_numeric(current_meep_scores, errors='coerce')

                    # Calculate AUC for Direct_MEEP
                    valid_mask = ~np.isnan(current_meep_scores) & ~np.isinf(current_meep_scores)
                    if np.sum(valid_mask) >= 2 and len(np.unique(labels[valid_mask])) >= 2:
                        auc = roc_auc_score(labels[valid_mask], current_meep_scores[valid_mask])
                    else:
                        auc = np.nan

                    # Calculate PMI Metrics for Direct_MEEP
                    pos_mask = valid_mask & (labels == 1)
                    if np.sum(pos_mask) >= 2:
                        p_scores = current_meep_scores[pos_mask]
                        p_pmi = true_pmi[pos_mask]
                        p_scores = z_normalize_scores(p_scores)
                        corr_spearman, _ = spearmanr(p_pmi, p_scores)
                        corr_pearson, _ = pearsonr(p_pmi, p_scores)
                        mse = mean_squared_error(p_pmi, p_scores)
                    else:
                        corr_spearman, corr_pearson, mse = np.nan, np.nan, np.nan

                    metrics_buffer.append({
                        "Dataset_Type": "Synthetic",
                        "Dataset": ds_name,
                        "Model": model_name,
                        "Method": "Direct_MEEP",
                        "Round": round_id,
                        "AUC": auc,
                        "Spearman_PMI": corr_spearman,
                        "Pearson_PMI": corr_pearson,
                        "MSE_PMI": mse
                    })
                else:
                    print(f"Warning: Direct_MEEP raw scores not available for {f_path}. Skipping.")

            # 3. Calculate metrics for other scoring methods
            method_cols = [c for c in df.columns if c.startswith("score_")]

            for method in method_cols:
                method_name = method.replace("score_", "")
                scores = pd.to_numeric(df[method].values, errors='coerce')

                # AUC
                valid_mask = ~np.isnan(scores) & ~np.isinf(scores)
                if np.sum(valid_mask) < 2 or len(np.unique(labels[valid_mask])) < 2:
                    continue

                v_scores = scores[valid_mask]
                v_labels = labels[valid_mask]

                try:
                    auc = roc_auc_score(v_labels, v_scores)
                except ValueError:
                    auc = np.nan

                # PMI Metrics
                pos_mask = valid_mask & (labels == 1)
                if np.sum(pos_mask) < 2:
                    corr_spearman, corr_pearson, mse = np.nan, np.nan, np.nan
                else:
                    p_scores = scores[pos_mask]
                    p_pmi = true_pmi[pos_mask]

                    # Z-normalize MEEP-based scores
                    if "MEEP" in method_name:
                        p_scores = z_normalize_scores(p_scores)

                    corr_spearman, _ = spearmanr(p_pmi, p_scores)
                    corr_pearson, _ = pearsonr(p_pmi, p_scores)
                    mse = mean_squared_error(p_pmi, p_scores)

                metrics_buffer.append({
                    "Dataset_Type": "Synthetic",
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Method": method_name,
                    "Round": round_id,
                    "AUC": auc,
                    "Spearman_PMI": corr_spearman,
                    "Pearson_PMI": corr_pearson,
                    "MSE_PMI": mse
                })

        except Exception as e:
            print(f"[Err] Processing {f_path}: {e}")

    return metrics_buffer


def process_empirical(ds_name, model_name, model_dir):
    """
    Process Empirical dataset results for a single model.

    Handles both test set (AUC) and human evaluation set (correlation).

    Args:
        ds_name: Dataset name (e.g., 'en', 'zh')
        model_name: Model name
        model_dir: Path to model results directory

    Returns:
        list: List of metric dictionaries for each round
    """
    metrics_buffer = []

    # Map round IDs to their inference files (test and human)
    files_map = {}
    for f in model_dir.glob("round_*_inference_results.csv"):
        parts = f.name.split("_")
        r_id = int(parts[1])
        split = parts[2]
        if r_id not in files_map:
            files_map[r_id] = {}
        files_map[r_id][split] = f

    # Load Direct MEEP scores and metadata for both test and human splits
    full_direct_scores_raw_test = load_direct_scores("empirical", ds_name, model_name, "test")
    full_direct_scores_raw_human = load_direct_scores("empirical", ds_name, model_name, "human")
    full_test_meta = load_full_meta("empirical", ds_name, model_name, "test")
    full_human_meta = load_full_meta("empirical", ds_name, model_name, "human")

    for r_id, paths in files_map.items():
        # --- Part A: Test Set (AUC) ---
        if "test" in paths:
            try:
                df = pd.read_csv(paths["test"])
                labels = df['is_positive'].values

                # Process Direct_MEEP
                if full_direct_scores_raw_test is not None and full_test_meta is not None:
                    indices_from_full_data = recreate_sampled_indices(
                        full_test_meta, r_id,
                        AggregationConfig.GROUP_SIZE, AggregationConfig.NEG_SAMPLES_USED
                    )

                    if len(indices_from_full_data) == len(df):
                        raw_scores_json_for_round = full_direct_scores_raw_test[indices_from_full_data]

                        target_idx = (r_id - 1) % AggregationConfig.NUM_MEEP_SCORES

                        current_meep_scores = []
                        for json_str in raw_scores_json_for_round:
                            try:
                                scores_list = json.loads(json_str)
                                if target_idx < len(scores_list):
                                    current_meep_scores.append(scores_list[target_idx])
                                else:
                                    current_meep_scores.append(np.nan)
                            except (json.JSONDecodeError, TypeError):
                                current_meep_scores.append(np.nan)

                        current_meep_scores = pd.to_numeric(current_meep_scores, errors='coerce')

                        valid_mask = ~np.isnan(current_meep_scores) & ~np.isinf(current_meep_scores)
                        if np.sum(valid_mask) >= 2 and len(np.unique(labels[valid_mask])) >= 2:
                            auc = roc_auc_score(labels[valid_mask], current_meep_scores[valid_mask])
                        else:
                            auc = np.nan

                        metrics_buffer.append({
                            "key": (ds_name, model_name, "Direct_MEEP", r_id),
                            "AUC": auc
                        })
                    else:
                        print(f"Warning: Direct_MEEP indices mismatch for {paths['test']}.")

                # Process other methods
                method_cols = [c for c in df.columns if c.startswith("score_")]
                for method in method_cols:
                    try:
                        scores = pd.to_numeric(df[method].values, errors='coerce')
                        valid_mask = ~np.isnan(scores) & ~np.isinf(scores)
                        if np.sum(valid_mask) < 2 or len(np.unique(labels[valid_mask])) < 2:
                            continue
                        auc = roc_auc_score(labels[valid_mask], scores[valid_mask])
                    except ValueError:
                        auc = np.nan

                    metrics_buffer.append({
                        "key": (ds_name, model_name, method.replace("score_", ""), r_id),
                        "AUC": auc
                    })
            except Exception as e:
                print(f"Err Empirical Test {ds_name} R{r_id}: {e}")

        # --- Part B: Human Set (Correlation) ---
        if "human" in paths:
            try:
                df = pd.read_csv(paths["human"])
                target_eng = df.get('annot_engaging_mean')
                target_rel = df.get('annot_relevant_mean')

                # Process Direct_MEEP
                if full_direct_scores_raw_human is not None and full_human_meta is not None:
                    indices_from_full_data = recreate_sampled_indices(
                        full_human_meta, r_id,
                        AggregationConfig.GROUP_SIZE, AggregationConfig.NEG_SAMPLES_USED
                    )

                    if len(indices_from_full_data) == len(df):
                        raw_scores_json_for_round = full_direct_scores_raw_human[indices_from_full_data]

                        target_idx = (r_id - 1) % AggregationConfig.NUM_MEEP_SCORES

                        current_meep_scores = []
                        for json_str in raw_scores_json_for_round:
                            try:
                                scores_list = json.loads(json_str)
                                if target_idx < len(scores_list):
                                    current_meep_scores.append(scores_list[target_idx])
                                else:
                                    current_meep_scores.append(np.nan)
                            except (json.JSONDecodeError, TypeError):
                                current_meep_scores.append(np.nan)

                        current_meep_scores = pd.to_numeric(current_meep_scores, errors='coerce')

                        res = {}
                        if target_eng is not None:
                            mask = ~np.isnan(current_meep_scores) & ~np.isinf(current_meep_scores) & ~np.isnan(target_eng)
                            res['Spearman_Engaging'] = spearmanr(target_eng[mask], current_meep_scores[mask])[0] if np.sum(mask) > 2 else np.nan
                            res['Pearson_Engaging'] = pearsonr(target_eng[mask], current_meep_scores[mask])[0] if np.sum(mask) > 2 else np.nan

                        if target_rel is not None:
                            mask = ~np.isnan(current_meep_scores) & ~np.isinf(current_meep_scores) & ~np.isnan(target_rel)
                            res['Spearman_Relevant'] = spearmanr(target_rel[mask], current_meep_scores[mask])[0] if np.sum(mask) > 2 else np.nan
                            res['Pearson_Relevant'] = pearsonr(target_rel[mask], current_meep_scores[mask])[0] if np.sum(mask) > 2 else np.nan

                        key = (ds_name, model_name, "Direct_MEEP", r_id)
                        found = False
                        for item in metrics_buffer:
                            if item.get("key") == key:
                                item.update(res)
                                found = True
                                break
                        if not found:
                            base = {"key": key, "AUC": np.nan}
                            base.update(res)
                            metrics_buffer.append(base)

                    else:
                        print(f"Warning: Direct_MEEP indices mismatch for {paths['human']}.")

                # Process other methods
                method_cols = [c for c in df.columns if c.startswith("score_")]
                for method in method_cols:
                    m_name = method.replace("score_", "")
                    scores = pd.to_numeric(df[method].values, errors='coerce')

                    res = {}
                    if target_eng is not None:
                        mask = ~np.isnan(scores) & ~np.isinf(scores) & ~np.isnan(target_eng)
                        res['Spearman_Engaging'] = spearmanr(target_eng[mask], scores[mask])[0] if np.sum(mask) > 2 else np.nan
                        res['Pearson_Engaging'] = pearsonr(target_eng[mask], scores[mask])[0] if np.sum(mask) > 2 else np.nan

                    if target_rel is not None:
                        mask = ~np.isnan(scores) & ~np.isinf(scores) & ~np.isnan(target_rel)
                        res['Spearman_Relevant'] = spearmanr(target_rel[mask], scores[mask])[0] if np.sum(mask) > 2 else np.nan
                        res['Pearson_Relevant'] = pearsonr(target_rel[mask], scores[mask])[0] if np.sum(mask) > 2 else np.nan

                    key = (ds_name, model_name, m_name, r_id)
                    found = False
                    for item in metrics_buffer:
                        if item.get("key") == key:
                            item.update(res)
                            found = True
                            break
                    if not found:
                        base = {"key": key, "AUC": np.nan}
                        base.update(res)
                        metrics_buffer.append(base)

            except Exception as e:
                print(f"Err Empirical Human {ds_name} R{r_id}: {e}")

    # Clean up and format results
    final_list = []
    for item in metrics_buffer:
        if "key" in item:
            (d, m, met, r) = item["key"]
            del item["key"]
            item["Dataset_Type"] = "Empirical"
            item["Dataset"] = d
            item["Model"] = m
            item["Method"] = met
            item["Round"] = r
            final_list.append(item)
    return final_list


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main aggregation function that processes all datasets and saves results."""
    all_metrics = []

    print(f"{'='*20} Starting Analysis {'='*20}")

    # 1. Process Synthetic Datasets
    syn_dir = AggregationConfig.RESULT_DIR / "synthetic"
    if syn_dir.exists():
        for ds_path in syn_dir.iterdir():
            if ds_path.name in AggregationConfig.SYN_DATASETS:
                for model_path in ds_path.iterdir():
                    print(f"Processing Synthetic: {ds_path.name} - {model_path.name}")
                    all_metrics.extend(process_synthetic(ds_path.name, model_path.name, model_path))

    # 2. Process Empirical Datasets
    emp_dir = AggregationConfig.RESULT_DIR / "empirical"
    if emp_dir.exists():
        for ds_path in emp_dir.iterdir():
            if ds_path.name in AggregationConfig.EMP_DATASETS:
                for model_path in ds_path.iterdir():
                    print(f"Processing Empirical: {ds_path.name} - {model_path.name}")
                    all_metrics.extend(process_empirical(ds_path.name, model_path.name, model_path))

    if not all_metrics:
        print("No results found.")
        return

    # 3. Aggregation
    df_raw = pd.DataFrame(all_metrics)

    # Save raw round-level data (required by visualize_and_report.py)
    df_raw.to_csv(AggregationConfig.OUTPUT_DIR / "all_results_raw.csv", index=False)
    print(f"\nSaved raw results to {AggregationConfig.OUTPUT_DIR / 'all_results_raw.csv'}")

    # Group by (Dataset, Model, Method) and calculate Mean/Std
    group_cols = ["Dataset_Type", "Dataset", "Model", "Method"]
    numeric_cols = [c for c in df_raw.columns if c not in group_cols + ["Round"]]

    df_agg = df_raw.groupby(group_cols)[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten columns
    new_cols = []
    for c in df_agg.columns:
        if c[1] == '':
            new_cols.append(c[0])
        else:
            new_cols.append(f"{c[0]}_{c[1]}")
    df_agg.columns = new_cols

    # Save aggregated data
    df_agg.to_csv(AggregationConfig.OUTPUT_DIR / "aggregated_results.csv", index=False)
    print(f"Saved aggregated results to {AggregationConfig.OUTPUT_DIR / 'aggregated_results.csv'}")

    # 4. Generate Display Tables

    # Table 1: Synthetic Performance
    print(f"\n\n>>> Synthetic Data Performance (AUC & PMI Metrics) <<<")
    syn_df = df_agg[df_agg['Dataset_Type'] == "Synthetic"].copy()
    if not syn_df.empty:
        for m in ["AUC", "Spearman_PMI", "Pearson_PMI", "MSE_PMI"]:
            if f'{m}_mean' in syn_df.columns and f'{m}_std' in syn_df.columns:
                syn_df[m] = syn_df.apply(
                    lambda x: f"{x[f'{m}_mean']:.3f} ({x[f'{m}_std']:.3f})" if pd.notnull(x[f'{m}_mean']) else "-",
                    axis=1
                )
            else:
                syn_df[m] = "-"

        display_cols = ["Dataset", "Model", "Method", "AUC", "Spearman_PMI", "Pearson_PMI", "MSE_PMI"]
        existing_display_cols = [col for col in display_cols if col in syn_df.columns]
        print(syn_df[existing_display_cols].to_markdown(index=False))
        syn_df[existing_display_cols].to_csv(AggregationConfig.OUTPUT_DIR / "table_synthetic.csv", index=False)

    # Table 2: Empirical Performance
    print(f"\n\n>>> Empirical Data Performance (AUC & Human Correlation) <<<")
    emp_df = df_agg[df_agg['Dataset_Type'] == "Empirical"].copy()
    if not emp_df.empty:
        for m in ["AUC", "Spearman_Engaging", "Pearson_Engaging", "Spearman_Relevant", "Pearson_Relevant"]:
            if f'{m}_mean' in emp_df.columns and f'{m}_std' in emp_df.columns:
                emp_df[m] = emp_df.apply(
                    lambda x: f"{x[f'{m}_mean']:.3f} ({x[f'{m}_std']:.3f})" if pd.notnull(x[f'{m}_mean']) else "-",
                    axis=1
                )
            else:
                emp_df[m] = "-"

        display_cols = ["Dataset", "Model", "Method", "AUC", "Spearman_Engaging", "Pearson_Engaging", "Spearman_Relevant", "Pearson_Relevant"]
        existing_display_cols = [col for col in display_cols if col in emp_df.columns]

        print(emp_df[existing_display_cols].to_markdown(index=False))
        emp_df[existing_display_cols].to_csv(AggregationConfig.OUTPUT_DIR / "table_empirical.csv", index=False)

    print(f"\nAll results saved to: {AggregationConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        os.system("pip install tabulate")

    main()
