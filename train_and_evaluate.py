"""
Module 3: Training and Evaluation of Scoring Heads

This module trains and evaluates scoring heads (Neural Networks and KDE estimators)
for dialogue response quality assessment using PMI-based approaches.

Training Methods:
- Neural Networks: PMI_NCE, MINE, InfoNCE (with Pair and Prompt modes)
- KDE Estimators: Pair Difference, Single Difference

The training process:
1. Load embeddings from Module 2
2. For T=5 rounds, sample different negative samples
3. Train models and perform inference on test/human splits
4. Save trained models and inference results
"""

import os
import json
import math
import pickle
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import io
import contextlib

# Import configuration from config module
import config


# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================

def load_split_data(ds_type, ds_name, model_name, split):
    """
    Load embeddings and metadata for a given split.

    Args:
        ds_type: Dataset type ('synthetic' or 'empirical')
        ds_name: Dataset name (e.g., 'diagonal', 'independent', 'en', 'zh')
        model_name: Model name (last component of model path)
        split: Split name ('train', 'val', 'test', 'human')

    Returns:
        tuple: (data_dict, meta_df) or (None, None) if not found
        data_dict: Dictionary with keys 'ctx', 'rsp', 'pmt' (embeddings)
        meta_df: DataFrame with metadata
    """
    base = config.Config.EMBEDDING_DIR / ds_type / ds_name / model_name
    files = {
        "ctx": base / f"{split}_context_embeddings.npy",
        "rsp": base / f"{split}_response_embeddings.npy",
        "pmt": base / f"{split}_prompt_embeddings.npy",
        "meta": base / f"{split}_meta.csv"
    }

    if not files["meta"].exists():
        return None, None

    try:
        meta = pd.read_csv(files["meta"])
        data = {}
        if files["ctx"].exists():
            data["ctx"] = np.load(files["ctx"])
        if files["rsp"].exists():
            data["rsp"] = np.load(files["rsp"])
        if files["pmt"].exists():
            data["pmt"] = np.load(files["pmt"])
        return data, meta
    except Exception as e:
        print(f"Error loading {ds_name}/{split}: {e}")
        return None, None


def sample_negatives(data_dict, meta_df, round_num):
    """
    Sample negative samples for a specific round.

    From each group of 16 samples (1 positive + 15 negatives), keep:
    - 1 positive sample (always first)
    - 3 negative samples determined by round_num

    This ensures each round uses different negative samples for diversity.

    Args:
        data_dict: Dictionary of embeddings
        meta_df: Metadata DataFrame
        round_num: Round number (1-5)

    Returns:
        tuple: (sampled_data_dict, sampled_meta_df)
    """
    total_samples = len(meta_df)
    if total_samples % config.Config.GROUP_SIZE != 0:
        total_samples = (total_samples // config.Config.GROUP_SIZE) * config.Config.GROUP_SIZE

    indices_to_keep = []

    for i in range(0, total_samples, config.Config.GROUP_SIZE):
        # Positive sample is always the first one in the group
        indices_to_keep.append(i)

        # Calculate which 3 negatives to keep based on round_num
        start_relative_neg_idx_in_group = (round_num - 1) * config.Config.NEG_SAMPLES_USED + 1
        end_relative_neg_idx_in_group = round_num * config.Config.NEG_SAMPLES_USED + 1

        for j in range(start_relative_neg_idx_in_group, end_relative_neg_idx_in_group):
            indices_to_keep.append(i + j)

    indices_to_keep = np.array(indices_to_keep)
    indices_to_keep.sort()

    # Sample embeddings
    new_data = {}
    for k, v in data_dict.items():
        if v is not None:
            new_data[k] = v[indices_to_keep]

    # Sample metadata
    new_meta = meta_df.iloc[indices_to_keep].reset_index(drop=True)

    return new_data, new_meta


def get_input_tensor(data_dict, mode):
    """
    Prepare input tensor based on mode.

    Args:
        data_dict: Dictionary of embeddings ('ctx', 'rsp', 'pmt')
        mode: Input mode ('pair' or 'single')

    Returns:
        numpy.ndarray: Concatenated or single embeddings
    """
    if mode == 'pair':
        if 'ctx' not in data_dict or 'rsp' not in data_dict:
            return None
        return np.hstack([data_dict['ctx'], data_dict['rsp']])
    elif mode == 'single':
        return data_dict.get('pmt')
    return None


# ==========================================
# KDE ESTIMATORS
# ==========================================

class KDEPMIEstimator:
    """
    KDE-based PMI estimator with PCA and optional scaling.

    Estimates the log-ratio of positive and negative densities:
        score(x) = log p_pos(x) - log p_neg(x)

    Uses Scott's rule for automatic bandwidth selection.
    """

    def __init__(self, bandwidth="auto", pca_dim=128, scale=True):
        self.bandwidth = bandwidth
        self.pca_dim = pca_dim
        self.scale = scale
        self.scaler = None
        self.pca = None
        self.kde_pos = None
        self.kde_neg = None
        self._fitted = False

    @staticmethod
    def _auto_bw(n, d):
        """
        Scott's rule for automatic bandwidth selection.

        Args:
            n: Number of samples
            d: Dimensionality

        Returns:
            float: Bandwidth value
        """
        return (n * (d + 2) / 4.0) ** (-1.0 / (d + 4)) * np.sqrt(d)

    def _prep_xy(self, X):
        """Prepare input X by scaling and PCA transformation."""
        if self.scale:
            X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def fit(self, train_X, train_y):
        """
        Fit KDE estimator on training data.

        Args:
            train_X: Training embeddings
            train_y: Binary labels (0 for negative, 1 for positive)

        Returns:
            self: Fitted estimator
        """
        pos = train_X[train_y > 0.5]
        neg = train_X[train_y < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return self

        X_all = np.vstack([pos, neg])

        # Fit scaler
        if self.scale:
            self.scaler = StandardScaler().fit(X_all)

        # Fit PCA
        if self.pca_dim is not None:
            k = min(self.pca_dim, X_all.shape[1], X_all.shape[0])
            if k < X_all.shape[1]:
                X_all_t = self.scaler.transform(X_all) if self.scale else X_all
                self.pca = PCA(n_components=k).fit(X_all_t)

        # Transform positive and negative samples
        pos_t = self._prep_xy(pos)
        neg_t = self._prep_xy(neg)

        # Determine bandwidth
        if self.bandwidth == "auto":
            d = pos_t.shape[1]
            bw_p = self._auto_bw(len(pos_t), d)
            bw_n = self._auto_bw(len(neg_t), d)
        else:
            bw_p = bw_n = float(self.bandwidth)

        # Fit KDEs
        self.kde_pos = KernelDensity(bandwidth=max(1e-3, bw_p)).fit(pos_t)
        self.kde_neg = KernelDensity(bandwidth=max(1e-3, bw_n)).fit(neg_t)
        self._fitted = True

        return self

    def score(self, X, batch_size=2048):
        """
        Compute PMI scores for input embeddings.

        Args:
            X: Input embeddings
            batch_size: Batch size for inference

        Returns:
            numpy.ndarray: PMI scores (log p_pos - log p_neg)
        """
        if not self._fitted:
            return np.zeros(len(X))

        Xt = self._prep_xy(X)

        n = len(Xt)
        scores_pos = []
        scores_neg = []

        for i in tqdm(range(0, n, batch_size), desc="KDE Score", leave=False):
            batch = Xt[i : i + batch_size]
            scores_pos.append(self.kde_pos.score_samples(batch))
            scores_neg.append(self.kde_neg.score_samples(batch))

        full_pos = np.concatenate(scores_pos)
        full_neg = np.concatenate(scores_neg)

        return (full_pos - full_neg).astype(np.float32)


class SimpleDensityEstimator:
    """
    Simple KDE for marginal density estimation with PCA.

    Used for hybrid PMI estimation (log p(z) - log p(c) - log p(r)).
    """

    def __init__(self, pca_dim=128):
        self.pca_dim = pca_dim
        self.scaler = StandardScaler()
        self.pca = None
        self.kde = None

    def fit(self, X):
        """
        Fit density estimator on data.

        Args:
            X: Input embeddings

        Returns:
            self: Fitted estimator
        """
        X_scaled = self.scaler.fit_transform(X)
        n = min(self.pca_dim, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n)
        X_pca = self.pca.fit_transform(X_scaled)

        # Automatic bandwidth
        n_s, n_f = X_pca.shape
        bw = (n_s * (n_f + 2) / 4.)**(-1. / (n_f + 4))
        self.kde = KernelDensity(bandwidth=bw).fit(X_pca)

        return self

    def score_samples(self, X, batch_size=2048):
        """
        Compute log-density for input samples.

        Args:
            X: Input embeddings
            batch_size: Batch size for inference

        Returns:
            numpy.ndarray: Log-density scores
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        n = len(X_pca)
        scores = []
        for i in tqdm(range(0, n, batch_size), desc="SimpleKDE Score", leave=False):
            batch = X_pca[i : i + batch_size]
            scores.append(self.kde.score_samples(batch))

        return np.concatenate(scores)


def fit_kde_estimators(data_dict, meta_df, mode):
    """
    Fit KDE estimators based on specified mode.

    Args:
        data_dict: Dictionary of embeddings
        meta_df: Metadata DataFrame with 'is_positive' column
        mode: Estimation mode ('pair_diff', 'single_diff', 'hybrid_marginal')

    Returns:
        Fitted estimator(s)
    """
    y = meta_df['is_positive'].values.astype(float)

    if mode == 'pair_diff':
        X = np.hstack([data_dict['ctx'], data_dict['rsp']])
        est = KDEPMIEstimator(bandwidth=config.Config.KDE_BANDWIDTH, pca_dim=config.Config.KDE_PCA_DIM)
        est.fit(X, y)
        return est

    elif mode == 'single_diff':
        X = data_dict['pmt']
        est = KDEPMIEstimator(bandwidth=config.Config.KDE_BANDWIDTH, pca_dim=config.Config.KDE_PCA_DIM)
        est.fit(X, y)
        return est

    elif mode == 'hybrid_marginal':
        # Hybrid approach: PMI = log p(z) - log p(c) - log p(r)
        pos_mask = (y == 1)
        estimators = {}
        estimators['joint_z'] = SimpleDensityEstimator(config.Config.KDE_PCA_DIM).fit(data_dict['pmt'][pos_mask])
        estimators['marg_c'] = SimpleDensityEstimator(config.Config.KDE_PCA_DIM).fit(data_dict['ctx'][pos_mask])
        estimators['marg_r'] = SimpleDensityEstimator(config.Config.KDE_PCA_DIM).fit(data_dict['rsp'][pos_mask])
        return estimators

    return np.zeros(len(data_dict['ctx']))


def infer_kde(estimator, data_dict, mode):
    """
    Run KDE inference and return scores.

    Args:
        estimator: Fitted KDE estimator(s)
        data_dict: Dictionary of embeddings
        mode: Estimation mode

    Returns:
        numpy.ndarray: Computed scores
    """
    if mode == 'pair_diff':
        X = np.hstack([data_dict['ctx'], data_dict['rsp']])
        return estimator.score(X, batch_size=config.Config.KDE_INFER_BATCH_SIZE)

    elif mode == 'single_diff':
        X = data_dict['pmt']
        return estimator.score(X, batch_size=config.Config.KDE_INFER_BATCH_SIZE)

    elif mode == 'hybrid_marginal':
        # PMI = log p(z) - log p(c) - log p(r)
        log_z = estimator['joint_z'].score_samples(data_dict['pmt'], batch_size=config.Config.KDE_INFER_BATCH_SIZE)
        log_c = estimator['marg_c'].score_samples(data_dict['ctx'], batch_size=config.Config.KDE_INFER_BATCH_SIZE)
        log_r = estimator['marg_r'].score_samples(data_dict['rsp'], batch_size=config.Config.KDE_INFER_BATCH_SIZE)
        return log_z - log_c - log_r

    return np.zeros(len(data_dict['ctx']))


# ==========================================
# NEURAL NETWORKS
# ==========================================

class ScoringHead(nn.Module):
    """
    3-layer MLP scoring head for dialogue response quality assessment.

    Architecture:
        Input -> Linear(256) -> PReLU -> Linear(128) -> PReLU -> Linear(1) -> Output
    """

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.PReLU(),
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Tanh scaling to keep outputs bounded
        return 20.0 * torch.tanh(self.net(x) / 20.0)


class MEEPTrainingDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for training with positive and negative samples.

    Each sample contains:
    - 1 positive embedding
    - K negative embeddings (K = NEG_SAMPLES_USED = 3)
    """

    def __init__(self, X_all, is_positive_labels, neg_samples_per_pos, dim):
        self.pos_samples = []
        self.neg_samples = []
        self.dim = dim

        i = 0
        while i < len(is_positive_labels):
            if is_positive_labels[i] == 1:
                pos_emb = X_all[i]
                current_neg_embs = []

                # Extract K negative samples
                for j in range(1, neg_samples_per_pos + 1):
                    if i + j < len(is_positive_labels) and is_positive_labels[i + j] == 0:
                        current_neg_embs.append(X_all[i + j])
                    else:
                        print(f"Warning: Not enough negative samples for positive at index {i}. "
                              f"Expected {neg_samples_per_pos}, got {len(current_neg_embs)}")
                        break

                if len(current_neg_embs) == neg_samples_per_pos:
                    self.pos_samples.append(pos_emb)
                    self.neg_samples.append(np.array(current_neg_embs))
                i += (1 + neg_samples_per_pos)
            else:
                i += 1

    def __len__(self):
        return len(self.pos_samples)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.pos_samples[idx]),
                torch.FloatTensor(self.neg_samples[idx]))


def train_nn_head(X_all_embeddings, all_meta, loss_type, name="Model", device='cuda'):
    """
    Train a scoring head with specified loss function.

    Args:
        X_all_embeddings: Training embeddings
        all_meta: Metadata DataFrame with 'is_positive' column
        loss_type: Loss type ('pmi_nce', 'mine', 'infonce')
        name: Model name for progress bar
        device: Device to train on

    Returns:
        Trained PyTorch model
    """
    dim = X_all_embeddings.shape[1]
    model = ScoringHead(dim).to(device)

    # Scale learning rate based on embedding dimension
    optimizer = optim.AdamW(model.parameters(), lr=config.Config.LR * (1024.0 / dim))

    train_dataset = MEEPTrainingDataset(
        X_all_embeddings,
        all_meta['is_positive'].values,
        config.Config.NEG_SAMPLES_USED,
        dim
    )
    train_loader = DataLoader(train_dataset, batch_size=config.Config.BATCH_SIZE, shuffle=True)

    model.train()
    pbar = tqdm(range(config.Config.EPOCHS), desc=f"Train {name}", leave=False, unit="ep")

    for _ in pbar:
        epoch_loss, steps = 0, 0

        for xp_batch, xn_batch in train_loader:
            xp_batch, xn_batch = xp_batch.to(device), xn_batch.to(device)
            B, K = xp_batch.shape[0], xn_batch.shape[1]

            # Score positive samples
            sp = model(xp_batch)

            if loss_type == 'infonce':
                # InfoNCE loss: maximize score of positive vs negatives
                sn = model(xn_batch.view(-1, dim)).view(B, K)
                logits = torch.cat([sp, sn], dim=1)
                loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=device))
            else:
                xn_flat = xn_batch.view(-1, dim)
                sn = model(xn_flat)

                if loss_type == 'pmi_nce':
                    # PMI-NCE loss: E[log p_pos] - E[log(1 + sum p_neg)]
                    loss = -(sp.mean() - torch.exp(sn).mean())
                elif loss_type == 'mine':
                    # MINE loss: E[log p_pos] - log(E[exp(p_neg)])
                    loss = -(sp.mean() - (torch.logsumexp(sn, dim=0) - math.log(sn.size(0))))
                else:
                    raise ValueError(f"Unknown loss: {loss_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        if steps:
            pbar.set_postfix({'loss': f"{epoch_loss/steps:.4f}"})

    return model


def infer_nn(model, X, device='cuda'):
    """
    Run NN inference and return scores.

    Args:
        model: Trained PyTorch model
        X: Input embeddings
        device: Device to run inference on

    Returns:
        numpy.ndarray: Predicted scores
    """
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X)),
        batch_size=config.Config.BATCH_SIZE * 100,
        shuffle=False
    )
    scores = []

    with torch.no_grad():
        for (b,) in tqdm(loader, desc="NN Infer", leave=False):
            scores.append(model(b.to(device)).cpu().numpy())

    return np.concatenate(scores).flatten()


# ==========================================
# MAIN EXECUTION
# ==========================================

def check_round_complete(out_dir, round_num, ds_type):
    """
    Check if a training round is complete (all models and inference results exist).
    
    Args:
        out_dir: Output directory for this model
        round_num: Round number (1-5)
        ds_type: Dataset type ('synthetic' or 'empirical')
    
    Returns:
        bool: True if round is complete, False otherwise
    """
    splits = ["test"]
    if ds_type == "empirical":
        splits.append("human")
    
    for name, _, _ in config.Config.NN_CONFIGS:
        model_path = out_dir / f"round_{round_num}_{name}_model.pt"
        if not model_path.exists():
            return False
    
    for name, _ in config.Config.KDE_CONFIGS:
        pkl_path = out_dir / f"round_{round_num}_{name}_model.pkl"
        if not pkl_path.exists():
            return False
    
    for split in splits:
        result_path = out_dir / f"round_{round_num}_{split}_inference_results.csv"
        if not result_path.exists():
            return False
    
    return True


def main(force_overwrite=False, train_only=False, infer_only=False, models_filter=None):
    """
    Main function to train and evaluate all scoring heads.
    
    Args:
        force_overwrite: If True, retrain all models even if they exist.
        train_only: If True, only train models without inference.
        infer_only: If True, only run inference using existing models.
        models_filter: List of model names to process (None = all models).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Force Overwrite: {force_overwrite}")
    print(f"Train Only: {train_only}, Infer Only: {infer_only}")

    config.STATS.start("Module 3 Total")

    # Check if embedding directory exists
    if not config.Config.EMBEDDING_DIR.exists():
        print("[Error] Embedding directory not found. Please run Module 2 first.")
        config.STATS.end("Module 3 Total")
        return

    # Iterate over dataset types (synthetic, empirical)
    ds_types = [d.name for d in config.Config.EMBEDDING_DIR.iterdir() if d.is_dir()]

    for ds_type in ds_types:
        ds_names = [d.name for d in (config.Config.EMBEDDING_DIR / ds_type).iterdir() if d.is_dir()]

        for ds_name in ds_names:
            ds_path = config.Config.EMBEDDING_DIR / ds_type / ds_name
            model_dirs = [d for d in ds_path.iterdir() if d.is_dir()]
            
            for model_dir in model_dirs:
                model_name_for_path = model_dir.name
                emb_path = model_dir
                
                if models_filter and model_name_for_path not in models_filter:
                    continue

                current_model_task_name = f"M3-{ds_name}-{model_name_for_path}"
                config.STATS.start(current_model_task_name)

                print(f"\n{'='*60}")
                print(f"Processing: {ds_type}/{ds_name} | {model_name_for_path}")
                print(f"{'='*60}")

                # Create output directory
                out_dir = config.Config.RESULT_DIR / ds_type / ds_name / model_name_for_path
                out_dir.mkdir(parents=True, exist_ok=True)

                # Load training and validation data
                tr_data_full, tr_meta_full = load_split_data(ds_type, ds_name, model_name_for_path, "train")
                val_data, val_meta = load_split_data(ds_type, ds_name, model_name_for_path, "val")

                if tr_data_full is None:
                    print(f"[Skip] No training data found for {ds_name}/{model_name_for_path}")
                    config.STATS.end(current_model_task_name)
                    continue

                # =========================================
                # LOOP OVER T=5 ROUNDS
                # =========================================
                for t in range(1, config.Config.NUM_ROUNDS + 1):
                    if not force_overwrite and check_round_complete(out_dir, t, ds_type):
                        print(f"[Skip] Round {t} already complete for {model_name_for_path}")
                        continue
                    
                    print(f"\n>>> Round {t}/{config.Config.NUM_ROUNDS} <<<")

                    # A. Sample training data for this round
                    tr_data_sampled, tr_meta_sampled = sample_negatives(tr_data_full, tr_meta_full, t)

                    # B. Train Neural Networks
                    trained_nns = {}
                    if not infer_only:
                        nn_pbar = tqdm(config.Config.NN_CONFIGS, desc="Training NNs", leave=True)

                        for name, loss, in_mode in nn_pbar:
                            timer_name = f"R1-Train-{ds_name}-{model_name_for_path}-{name}"
                            if t == 1:
                                config.STATS.start(timer_name)

                            model_save_path = out_dir / f"round_{t}_{name}_model.pt"

                            X_tr = get_input_tensor(tr_data_sampled, in_mode)
                            if X_tr is None:
                                if t == 1:
                                    config.STATS.end(timer_name)
                                continue

                            dim = X_tr.shape[1]
                            model = ScoringHead(dim).to(device)

                            should_train = force_overwrite or not model_save_path.exists()
                            
                            if model_save_path.exists() and not force_overwrite:
                                model.load_state_dict(torch.load(model_save_path, map_location=device))
                            else:
                                model = train_nn_head(X_tr, tr_meta_sampled, loss, name=name, device=device)
                                torch.save(model.state_dict(), model_save_path)

                            trained_nns[name] = (model, in_mode)

                            if t == 1:
                                config.STATS.end(timer_name)
                    else:
                        for name, loss, in_mode in config.Config.NN_CONFIGS:
                            model_save_path = out_dir / f"round_{t}_{name}_model.pt"
                            if model_save_path.exists():
                                X_tr = get_input_tensor(tr_data_sampled, in_mode)
                                if X_tr is not None:
                                    dim = X_tr.shape[1]
                                    model = ScoringHead(dim).to(device)
                                    model.load_state_dict(torch.load(model_save_path, map_location=device))
                                    trained_nns[name] = (model, in_mode)

                    # C. Fit KDE Estimators
                    trained_kdes = {}
                    if not infer_only:
                        kde_pbar = tqdm(config.Config.KDE_CONFIGS, desc="Fitting KDEs", leave=True)

                        for name, mode in kde_pbar:
                            timer_name = f"R1-Fit-{ds_name}-{model_name_for_path}-{name}"
                            if t == 1:
                                config.STATS.start(timer_name)

                            pkl_save_path = out_dir / f"round_{t}_{name}_model.pkl"

                            should_train = force_overwrite or not pkl_save_path.exists()
                            
                            if pkl_save_path.exists() and not force_overwrite:
                                try:
                                    with open(pkl_save_path, "rb") as f:
                                        estimator = pickle.load(f)
                                    trained_kdes[name] = (estimator, mode)
                                except Exception as e:
                                    print(f"    [Warn] Could not load pre-trained KDE {name} from {pkl_save_path}: {e}")
                            else:
                                try:
                                    estimator = fit_kde_estimators(tr_data_sampled, tr_meta_sampled, mode)
                                    trained_kdes[name] = (estimator, mode)
                                    with open(pkl_save_path, "wb") as f:
                                        pickle.dump(estimator, f)
                                except Exception as e:
                                    print(f"    [Err] {name}: {e}")

                            if t == 1:
                                config.STATS.end(timer_name)
                    else:
                        for name, mode in config.Config.KDE_CONFIGS:
                            pkl_save_path = out_dir / f"round_{t}_{name}_model.pkl"
                            if pkl_save_path.exists():
                                try:
                                    with open(pkl_save_path, "rb") as f:
                                        estimator = pickle.load(f)
                                    trained_kdes[name] = (estimator, mode)
                                except Exception:
                                    pass

                    # D. Inference (skip if train_only)
                    if train_only:
                        continue
                        
                    splits = ["test"]
                    if ds_type == "empirical":
                        splits.append("human")

                    for split in splits:
                        te_data, te_meta = load_split_data(ds_type, ds_name, model_name_for_path, split)
                        if te_data is None:
                            continue

                        # Sample test data for this round (1 pos + 3 neg)
                        te_data_sampled, te_meta_sampled = sample_negatives(te_data, te_meta, t)

                        res_df = te_meta_sampled.copy()
                        drop_cols = [c for c in res_df.columns if 'context' in c or 'response' in c or 'text' in c]
                        res_df.drop(columns=drop_cols, inplace=True, errors='ignore')

                        # NN Inference
                        for name, (model, in_mode) in trained_nns.items():
                            timer_name = f"R1-Infer-{ds_name}-{model_name_for_path}-{name}-{split}"
                            if t == 1:
                                config.STATS.start(timer_name)

                            try:
                                X_in = get_input_tensor(te_data_sampled, in_mode)
                                if X_in is not None:
                                    res_df[f"score_{name}"] = infer_nn(model, X_in, device)
                            except Exception as e:
                                print(f"    [Err] NN inference for {name} on {split}: {e}")

                            if t == 1:
                                config.STATS.end(timer_name)

                        # KDE Inference
                        for name, (est, mode) in trained_kdes.items():
                            timer_name = f"R1-Infer-{ds_name}-{model_name_for_path}-{name}-{split}"
                            if t == 1:
                                config.STATS.start(timer_name)

                            try:
                                res_df[f"score_{name}"] = infer_kde(est, te_data_sampled, mode)
                            except Exception as e:
                                print(f"    [Err] KDE inference for {name} on {split}: {e}")

                            if t == 1:
                                config.STATS.end(timer_name)

                        # Save inference results
                        save_name = out_dir / f"round_{t}_{split}_inference_results.csv"
                        res_df.to_csv(save_name, index=False)

                        # Print AUC summary for test split
                        if split == "test" and 'is_positive' in res_df.columns:
                            print(f"  -> Round {t} Test AUC Summary:")
                            labels = res_df['is_positive'].values
                            score_cols = [c for c in res_df.columns if c.startswith("score_")]
                            score_cols.sort()

                            for col in score_cols:
                                try:
                                    auc = roc_auc_score(labels, res_df[col])
                                    method_name = col.replace("score_", "")
                                    print(f"     | {method_name:<16}: {auc:.4f}")
                                except Exception as e:
                                    print(f"     [Err] AUC calculation for {col}: {e}")

                config.STATS.end(current_model_task_name)

    print("\n[Module 3 Finished] All rounds completed.")
    config.STATS.end("Module 3 Total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMIScore Module 3: Training and Evaluation")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Force retrain all models even if they exist")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train models without running inference")
    parser.add_argument("--infer-only", action="store_true",
                        help="Only run inference using existing models (skip training)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of model names to process (e.g., 'Qwen3-4B,Qwen3-8B')")
    args = parser.parse_args()
    
    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(",")]
    
    main(
        force_overwrite=args.force_overwrite,
        train_only=args.train_only,
        infer_only=args.infer_only,
        models_filter=models_filter
    )
    config.STATS.summary("execution_summary_module3.txt")
