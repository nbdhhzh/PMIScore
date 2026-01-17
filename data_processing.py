# -*- coding: utf-8 -*-
"""
PMIScore Module 1: Data Processing

This module handles:
1. Synthetic dataset generation (diagonal, independent, block modes)
2. Empirical dataset download and processing (DSTC11 data)
3. Data paraphrasing (optional, requires API key)
4. Negative sample generation
5. Train/validation/test split

Output: CSV files in datasets/synthetic/ and datasets/empirical/
"""

import os
import random
import zipfile
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from config import Config, ExecutionStats

# Initialize execution stats
STATS = ExecutionStats()


class SyntheticDataProcessor:
    """
    Process and generate synthetic datasets.
    
    Generates synthetic dialogue datasets with controlled joint probability
    distributions to create controlled PMI scenarios.
    """
    
    def __init__(self):
        """Initialize SyntheticDataProcessor with modes and prototypes."""
        self.modes = Config.SYNTHETIC_CASES
        self.prototypes = {
            "diagonal": {
                "X": [
                    f"I feel {mood} today."
                    for mood in ["happy", "sad", "angry", "nervous", "relaxed",
                                "bored", "upset", "calm", "excited", "tired",
                                "anxious", "joyful", "disappointed", "satisfied",
                                "sleepy", "thrilled", "shy", "lonely", "peaceful",
                                "confused"]
                ],
                "Y": [
                    f"Sounds like you are {mood}."
                    for mood in ["happy", "sad", "angry", "nervous", "relaxed",
                                "bored", "upset", "calm", "excited", "tired",
                                "anxious", "joyful", "disappointed", "satisfied",
                                "sleepy", "thrilled", "shy", "lonely", "peaceful",
                                "confused"]
                ]
            },
            "independent": {
                "X": [
                    f"Have you visited {place} recently?"
                    for place in ["New York", "London", "Paris", "Tokyo", "Berlin",
                                "Sydney", "Toronto", "Chicago", "Boston",
                                "Los Angeles", "Madrid", "Rome", "Beijing", "Seoul",
                                "San Francisco", "Shanghai", "Singapore", "Dublin",
                                "Amsterdam", "Dubai"]
                ],
                "Y": [
                    f"I think {food} tastes great."
                    for food in ["pizza", "sushi", "tacos", "pasta", "ice cream",
                                "ramen", "dumplings", "burgers", "curry", "steak",
                                "salad", "sandwiches", "seafood", "chocolate",
                                "pancakes", "fries", "barbecue", "soup", "bread",
                                "cheese"]
                ]
            },
            "block": {
                "X": (
                    [f"I want to travel to {city}." for city in ["New York", "London", "Paris", "Tokyo", "Berlin"]] +
                    [f"I am learning {subject}." for subject in ["math", "physics", "chemistry", "history", "biology"]] +
                    [f"I like playing {sport}." for sport in ["basketball", "soccer", "tennis", "badminton", "table tennis"]] +
                    [f"The weather is {weather} today." for weather in ["sunny", "rainy", "cloudy", "snowy", "windy"]]
                ),
                "Y": [
                    "That sounds like an exciting destination!", "Travel can be so enriching and fun.",
                    "I hope you have a wonderful trip!", "Planning ahead always makes trips better.",
                    "Exploring new places is always rewarding.", "Learning new things is always valuable.",
                    "That's great that you're expanding your knowledge.", "Keep up the good work with your studies!",
                    "Education opens so many doors.", "It's never too late to learn something new.",
                    "Staying active is so important for health.", "Sports are a great way to stay fit and have fun.",
                    "Regular exercise makes such a difference.", "It's wonderful that you enjoy being active.",
                    "Physical activity is great for both body and mind.", "Weather can really affect our mood and plans.",
                    "I hope the weather is nice for your activities.", "Different weather brings different opportunities.",
                    "It's always good to be prepared for any weather.", "Each type of weather has its own charm."
                ]
            }
        }
    
    def _random_marginal(self, size, alpha=1.0):
        """
        Generate random marginal distribution using Dirichlet.
        
        Args:
            size: Size of the distribution.
            alpha: Concentration parameter for Dirichlet distribution.
        
        Returns:
            numpy array of marginal probabilities.
        """
        return np.random.dirichlet([alpha] * size)
    
    def _generate_joint_matrix(self, size, mode="diagonal", alpha=1.0, noise=0.2):
        """
        Generate joint probability matrix based on specified mode.
        
        Args:
            size: Size of the joint matrix (size x size).
            mode: Type of correlation structure ('diagonal', 'independent', 'block').
            alpha: Concentration parameter for marginal distributions.
            noise: Amount of noise to add to the base structure.
        
        Returns:
            tuple: (joint_matrix, px, py) where px and py are marginal distributions.
        """
        px = self._random_marginal(size, alpha)
        py = self._random_marginal(size, alpha)
        
        if mode == "diagonal":
            base = np.diag(np.random.rand(size))
        elif mode == "block":
            base = np.zeros((size, size))
            block_size = size // 4
            for k in range(4):
                s, e = k * block_size, (k + 1) * block_size
                base[s:e, s:e] = np.random.rand(block_size, block_size)
        else:  # independent
            base = np.random.rand(size, size)
        
        mat = base
        if noise > 0:
            mat += np.random.rand(size, size) * noise * mat.mean()
        mat = mat * px[:, None] * py[None, :]
        mat /= mat.sum()
        
        px = mat.sum(axis=1)
        py = mat.sum(axis=0)
        return mat, px, py
    
    def _sample_pairs(self, joint_mat, px, py, X_list, Y_list, total_samples):
        """
        Sample (x, y) pairs from the joint distribution.
        
        Args:
            joint_mat: Joint probability matrix.
            px: Marginal distribution for X.
            py: Marginal distribution for Y.
            X_list: List of X prototypes.
            Y_list: List of Y prototypes.
            total_samples: Number of samples to generate.
        
        Returns:
            list of dictionaries containing sampled pairs and their probabilities.
        """
        mat = np.array(joint_mat, dtype=float)
        p = mat.flatten() / mat.sum()
        indices = np.random.choice(len(p), size=total_samples, replace=True, p=p)
        Y_len = len(Y_list)
        
        pairs = []
        for idx in indices:
            i, j = idx // Y_len, idx % Y_len
            pairs.append({
                "x_id": int(i),
                "y_id": int(j),
                "context": X_list[i],
                "response": Y_list[j],
                "p_xy": float(joint_mat[i, j]),
                "p_x": float(px[i]),
                "p_y": float(py[j])
            })
        return pairs
    
    def _paraphrase_single(self, text, style):
        """
        Paraphrase a single text with specified style (requires API key).
        
        Args:
            text: Text to paraphrase.
            style: Paraphrasing style (e.g., 'formal and professional').
        
        Returns:
            Paraphrased text, or original text if API key is not available.
        """
        if not Config.API_KEY:
            return text
        
        headers = {
            "Authorization": f"Bearer {Config.API_KEY}",
            "Content-Type": "application/json"
        }
        
        content = (
            f"Rewrite the text into style: '{style}'. Keep the original meaning unchanged.\n"
            "Output ONLY the rewritten text.\n"
            f"Rewrite: {text}\n"
            "Response: "
        )
        
        data = {
            "model": "google/gemini-2.5-flash-lite",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.8
        }
        
        try:
            resp = requests.post(Config.API_URL, headers=headers, json=data, timeout=15)
            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"].strip()
                if result.startswith("Paraphrased: "):
                    result = result[13:].strip()
                return result.strip('"').strip("' ")
        except Exception:
            pass
        return text
    
    def _paraphrase_dataset(self, data, desc_prefix=""):
        """
        Paraphrase a dataset (context and response for each sample).
        
        Args:
            data: List of dictionaries containing 'context' and 'response'.
            desc_prefix: Description prefix for progress bar.
        
        Returns:
            List of dictionaries with added 'context_paraphrased' and 'response_paraphrased'.
        """
        if not Config.API_KEY:
            for ex in data:
                ex["context_paraphrased"] = ex["context"]
                ex["response_paraphrased"] = ex["response"]
            return data
        
        print(f"[Info] Paraphrasing {len(data)} samples ({desc_prefix})...")
        context_results = [None] * len(data)
        response_results = [None] * len(data)
        
        # Pre-generate styles for reproducibility
        ctx_styles = [random.choice(Config.PARAPHRASE_STYLES) for _ in range(len(data))]
        rsp_styles = [random.choice(Config.PARAPHRASE_STYLES) for _ in range(len(data))]
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Submit context tasks
            ctx_futures = {
                executor.submit(self._paraphrase_single, ex["context"], ctx_styles[i]): i
                for i, ex in enumerate(data)
            }
            # Submit response tasks
            rsp_futures = {
                executor.submit(self._paraphrase_single, ex["response"], rsp_styles[i]): i
                for i, ex in enumerate(data)
            }
            
            # Process context results with tqdm
            for future in tqdm(as_completed(ctx_futures), total=len(data),
                            desc=f"  > Contexts ({desc_prefix})"):
                idx = ctx_futures[future]
                context_results[idx] = future.result() if not future.exception() else data[idx]["context"]
            
            # Process response results with tqdm
            for future in tqdm(as_completed(rsp_futures), total=len(data),
                            desc=f"  > Responses ({desc_prefix})"):
                idx = rsp_futures[future]
                response_results[idx] = future.result() if not future.exception() else data[idx]["response"]
        
        for i, ex in enumerate(data):
            ex["context_paraphrased"] = context_results[i]
            ex["response_paraphrased"] = response_results[i]
        return data
    
    def _generate_negatives(self, pos_data, joint_mat, px, py, desc_prefix=""):
        """
        Generate negative samples for each positive sample.
        
        Args:
            pos_data: DataFrame containing positive samples.
            joint_mat: Joint probability matrix.
            px: Marginal distribution for X.
            py: Marginal distribution for Y.
            desc_prefix: Description prefix for progress bar.
        
        Returns:
            DataFrame with positive and negative samples.
        """
        all_responses = pos_data['response'].tolist()
        all_y_ids = pos_data['y_id'].tolist()
        
        if 'response_paraphrased' in pos_data.columns:
            all_resp_para = pos_data['response_paraphrased'].tolist()
        else:
            all_resp_para = all_responses
        
        context_map = defaultdict(list)
        for idx, x_val in enumerate(pos_data['x_id']):
            context_map[int(x_val)].append(idx)
        
        final_data = []
        
        # Wrap iterrows with tqdm for progress bar
        iterator = tqdm(pos_data.iterrows(), total=pos_data.shape[0],
                       desc=f"  > Neg Sampling ({desc_prefix})")
        
        for i, row in iterator:
            pos_item = row.to_dict()
            pos_item['is_positive'] = 1
            final_data.append(pos_item)
            
            current_x_id = int(row['x_id'])
            current_resp = row['response']
            current_ctx = row.get('context', '')
            current_ctx_para = row.get('context_paraphrased', current_ctx)
            
            for _ in range(Config.TRAIN_NEG_RATIO):
                target_idx = random.randint(0, len(all_responses) - 1)
                neg_y_id = int(all_y_ids[target_idx])
                neg_item = {
                    'context': current_ctx,
                    'response': all_responses[target_idx],
                    'context_paraphrased': current_ctx_para,
                    'response_paraphrased': all_resp_para[target_idx],
                    'is_positive': 0,
                    'x_id': current_x_id,
                    'y_id': neg_y_id,
                    'p_xy': float(joint_mat[current_x_id, neg_y_id]),
                    'p_x': float(px[current_x_id]),
                    'p_y': float(py[neg_y_id])
                }
                final_data.append(neg_item)
        
        return pd.DataFrame(final_data)
    
    def run(self):
        """
        Main execution method for synthetic data processing.
        
        Generates synthetic datasets for all modes (diagonal, independent, block).
        """
        print(f"\n{'='*20} Processing Synthetic Data {'='*20}")
        
        mode_seeds = {
            "diagonal": 0,
            "independent": 1000,
            "block": 2000
        }
        
        for mode in self.modes:
            STATS.start(f"Syn - {mode} - Total")
            # Reset seed for each mode
            current_seed = Config.SEED + mode_seeds.get(mode, 0)
            Config.set_seed(current_seed)
            print(f"--- Processing {mode} (Seed: {current_seed}) ---")
            
            mode_dir = Config.SYNTHETIC_DIR / mode
            mode_dir.mkdir(exist_ok=True)
            
            joint, px, py = self._generate_joint_matrix(Config.SYN_SIZE, mode=mode)
            
            # Output file paths
            f_train = mode_dir / "train.csv"
            f_val = mode_dir / "val.csv"
            f_test = mode_dir / "test.csv"
            
            # Load or generate positive samples
            if f_train.exists() and f_val.exists() and f_test.exists():
                print(f"[Info] Positive samples already exist, loading directly...")
                train = pd.read_csv(f_train)
                val = pd.read_csv(f_val)
                test = pd.read_csv(f_test)
            else:
                STATS.start(f"Syn - {mode} - Gen & Para")
                print(f"[Info] Positive samples do not exist, generating and paraphrasing...")
                pairs = self._sample_pairs(
                    joint, px, py,
                    self.prototypes[mode]["X"],
                    self.prototypes[mode]["Y"],
                    Config.SYN_SAMPLES
                )
                
                df = pd.DataFrame(pairs)
                n = len(df)
                n_train = int(Config.SYN_SPLIT[0] * n)
                n_val = int(Config.SYN_SPLIT[1] * n)
                
                train = df.iloc[:n_train].copy()
                val = df.iloc[n_train:n_train + n_val].copy()
                test = df.iloc[n_train + n_val:].copy()
                
                # Paraphrase with progress bars
                train = pd.DataFrame(self._paraphrase_dataset(train.to_dict('records'), desc_prefix="train"))
                val = pd.DataFrame(self._paraphrase_dataset(val.to_dict('records'), desc_prefix="val"))
                test = pd.DataFrame(self._paraphrase_dataset(test.to_dict('records'), desc_prefix="test"))
                
                train.to_csv(f_train, index=False)
                val.to_csv(f_val, index=False)
                test.to_csv(f_test, index=False)
                STATS.end(f"Syn - {mode} - Gen & Para")
            
            # Generate negative samples
            f_train_neg = mode_dir / "train_with_negatives.csv"
            f_val_neg = mode_dir / "val_with_negatives.csv"
            f_test_neg = mode_dir / "test_with_negatives.csv"
            
            if f_train_neg.exists() and f_val_neg.exists() and f_test_neg.exists():
                print(f"[Info] Negative sample files already exist, skipping resampling.")
            else:
                STATS.start(f"Syn - {mode} - Neg Sampling")
                print(f"[Info] Resampling negative samples ({Config.TRAIN_NEG_RATIO}x)...")
                self._generate_negatives(train, joint, px, py, desc_prefix="train").to_csv(f_train_neg, index=False)
                self._generate_negatives(val, joint, px, py, desc_prefix="val").to_csv(f_val_neg, index=False)
                self._generate_negatives(test, joint, px, py, desc_prefix="test").to_csv(f_test_neg, index=False)
                STATS.end(f"Syn - {mode} - Neg Sampling")
            
            STATS.end(f"Syn - {mode} - Total")
            print(f"[Success] {mode} dataset updated.")


class EmpiricalDataProcessor:
    """
    Process empirical datasets from DSTC11 Track 4.
    
    Downloads and processes real dialogue data from DSTC11 for
    English and Chinese languages.
    """
    
    def __init__(self):
        """Initialize EmpiricalDataProcessor with language settings."""
        self.langs = Config.EMPIRICAL_LANGS
        self.human_eval_files = Config.HUMAN_EVAL_FILES
    
    def _download_data(self):
        """
        Download DSTC11 Track 4 data.
        
        Downloads the zip file from HuggingFace and extracts it locally.
        Skips download if the directory already exists.
        """
        STATS.start("Emp - Download")
        zip_name = "DSTC_11_Track_4.zip"
        target_dir = Path(Config.DSTC_DOWNLOAD_DIR)
        
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[Info] Directory {target_dir} already exists and is not empty, skipping download.")
            STATS.end("Emp - Download")
            return
        
        if not os.path.exists(zip_name):
            print("[Info] Downloading DSTC11 Data (via wget)....")
            result = os.system(f"wget -c -t 5 --show-progress -O {zip_name} {Config.DSTC_ZIP_URL}")
            if result != 0:
                raise RuntimeError(f"Download failed with exit code {result}")
        else:
            print(f"[Info] {zip_name} already exists, preparing to extract.")
        
        print("[Info] Extracting...")
        try:
            with zipfile.ZipFile(zip_name, "r") as zf:
                zf.extractall(".")
            print("[Info] Extraction complete.")
            os.remove(zip_name)
            print("[Info] Zip file removed to save space.")
        except zipfile.BadZipFile:
            print("[Error] Zip file is corrupted. Removing it. Please re-run.")
            os.remove(zip_name)
            raise
        STATS.end("Emp - Download")
    
    def _truncate_context(self, text, max_len=30000):
        """
        Truncate context to maximum length while preserving structure.
        
        Args:
            text: Context text to truncate.
            max_len: Maximum length in characters.
        
        Returns:
            Truncated context text.
        """
        if len(text) <= max_len:
            return text
        truncated = text[-max_len:]
        first_newline = truncated.find('\n')
        if first_newline != -1 and first_newline < 5000:
            return truncated[first_newline + 1:]
        return truncated
    
    def _process_dialogues(self, lang):
        """
        Process dialogue CSV files for a given language.
        
        Args:
            lang: Language code ('en' or 'zh').
        
        Returns:
            List of dialogue pairs with context and response.
        """
        target_dir = Path(Config.DSTC_DOWNLOAD_DIR) / "metadata" / "train" / lang
        if not target_dir.exists():
            print(f"[Warn] Directory not found: {target_dir}")
            return []
        
        files = list(target_dir.rglob("*_main.csv"))
        dialogs = defaultdict(list)
        
        print(f"Processing {lang} files: {len(files)}")
        for f in tqdm(files, desc=f"Reading {lang} CSVs"):
            try:
                df = pd.read_csv(f)
                if not {"UID", "SEG"}.issubset(df.columns):
                    continue
                for _, row in df.iterrows():
                    uid, seg = str(row["UID"]), str(row["SEG"]).strip()
                    if not seg or seg == "nan":
                        continue
                    parts = uid.split("-")
                    if len(parts) >= 2:
                        dialogs[f"{parts[0]}-{parts[1]}"].append(seg)
            except Exception:
                pass
        
        pairs = []
        for dkey, turns in dialogs.items():
            if len(turns) < 2:
                continue
            for i in range(1, len(turns)):
                current_turns = turns[:i][-Config.MAX_CONTEXT_TURNS:]
                ctx_str = "\n".join(current_turns).strip()
                ctx_str = self._truncate_context(ctx_str, Config.MAX_CONTEXT_CHARS)
                pairs.append({
                    "dialog_key": dkey,
                    "context": ctx_str,
                    "response": turns[i].strip()
                })
        
        if len(pairs) > Config.EMP_SAMPLES:
            pairs = random.sample(pairs, Config.EMP_SAMPLES)
        return pairs
    
    def _add_negatives(self, pos_pairs, desc_prefix=""):
        """
        Add negative samples to positive pairs.
        
        Args:
            pos_pairs: List of positive dialogue pairs.
            desc_prefix: Description prefix for progress bar.
        
        Returns:
            DataFrame with positive and negative samples.
        """
        if isinstance(pos_pairs, list):
            df = pd.DataFrame(pos_pairs)
        else:
            df = pos_pairs
        
        all_responses = df['response'].tolist()
        final_data = []
        
        iterator = tqdm(df.iterrows(), total=df.shape[0],
                       desc=f"  > Neg Sampling ({desc_prefix})")
        
        for i, row in iterator:
            pos_item = row.to_dict()
            pos_item['is_positive'] = 1
            pos_item['x_id'] = i
            pos_item['y_id'] = i
            pos_item['p_xy'] = 1.0
            pos_item['p_x'] = 1.0
            pos_item['p_y'] = 1.0
            final_data.append(pos_item)
            
            current_ctx = pos_item['context']
            
            for _ in range(Config.TRAIN_NEG_RATIO):
                target_idx = random.randint(0, len(all_responses) - 1)
                neg_item = {
                    'dialog_key': pos_item['dialog_key'],
                    'context': current_ctx,
                    'response': all_responses[target_idx],
                    'is_positive': 0,
                    'x_id': i,
                    'y_id': target_idx,
                    'p_xy': 0.0,
                    'p_x': 1.0,
                    'p_y': 1.0
                }
                final_data.append(neg_item)
        
        return pd.DataFrame(final_data)
    
    def _load_human_eval(self, lang):
        """
        Load human evaluation data for a given language.
        
        Args:
            lang: Language code ('en' or 'zh').
        
        Returns:
            DataFrame with human evaluation scores.
        """
        files = self.human_eval_files.get(lang, [])
        all_data = []
        
        for file_path in files:
            full_path = Path(Config.DSTC_DOWNLOAD_DIR) / file_path
            if not full_path.exists():
                continue
            try:
                import json
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
            except Exception as e:
                print(f"[Warn] Could not load human eval from {full_path}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Ensure required columns exist
            if 'context' not in df.columns and 'dialogue_context' in df.columns:
                df['context'] = df['dialogue_context']
            if 'response' not in df.columns and 'model_response' in df.columns:
                df['response'] = df['model_response']
            return df
        return None
    
    def run(self):
        """
        Main execution method for empirical data processing.
        
        Downloads and processes DSTC11 data for all languages.
        """
        print(f"\n{'='*20} Processing Empirical Data {'='*20}")
        
        # Download data
        self._download_data()
        
        for lang in self.langs:
            STATS.start(f"Emp - {lang} - Total")
            print(f"--- Processing {lang} ---")
            
            lang_dir = Config.EMPIRICAL_DIR / lang
            lang_dir.mkdir(exist_ok=True)
            
            # Output file paths
            f_train = lang_dir / "train.csv"
            f_val = lang_dir / "val.csv"
            f_test = lang_dir / "test.csv"
            f_human = lang_dir / "human.csv"
            
            # Process dialogues
            if f_train.exists() and f_val.exists() and f_test.exists():
                print(f"[Info] Data files already exist for {lang}, loading directly...")
                train = pd.read_csv(f_train)
                val = pd.read_csv(f_val)
                test = pd.read_csv(f_test)
            else:
                STATS.start(f"Emp - {lang} - Processing")
                print(f"[Info] Processing dialogues for {lang}...")
                pairs = self._process_dialogues(lang)
                
                if len(pairs) == 0:
                    print(f"[Warn] No dialogues found for {lang}")
                    STATS.end(f"Emp - {lang} - Processing")
                    STATS.end(f"Emp - {lang} - Total")
                    continue
                
                df = pd.DataFrame(pairs)
                n = len(df)
                n_train = int(Config.EMP_SPLIT[0] * n)
                n_val = int(Config.EMP_SPLIT[1] * n)
                
                train = df.iloc[:n_train].copy()
                val = df.iloc[n_train:n_train + n_val].copy()
                test = df.iloc[n_train + n_val:].copy()
                
                train.to_csv(f_train, index=False)
                val.to_csv(f_val, index=False)
                test.to_csv(f_test, index=False)
                STATS.end(f"Emp - {lang} - Processing")
            
            # Generate negative samples
            f_train_neg = lang_dir / "train_with_negatives.csv"
            f_val_neg = lang_dir / "val_with_negatives.csv"
            f_test_neg = lang_dir / "test_with_negatives.csv"
            
            if f_train_neg.exists() and f_val_neg.exists() and f_test_neg.exists():
                print(f"[Info] Negative sample files already exist for {lang}, skipping resampling.")
            else:
                STATS.start(f"Emp - {lang} - Neg Sampling")
                print(f"[Info] Resampling negative samples for {lang} ({Config.TRAIN_NEG_RATIO}x)...")
                self._add_negatives(train, desc_prefix="train").to_csv(f_train_neg, index=False)
                self._add_negatives(val, desc_prefix="val").to_csv(f_val_neg, index=False)
                self._add_negatives(test, desc_prefix="test").to_csv(f_test_neg, index=False)
                STATS.end(f"Emp - {lang} - Neg Sampling")
            
            # Load human evaluation data
            if f_human.exists():
                print(f"[Info] Human evaluation file already exists for {lang}, loading directly...")
            else:
                STATS.start(f"Emp - {lang} - Human Eval")
                print(f"[Info] Loading human evaluation data for {lang}...")
                human_df = self._load_human_eval(lang)
                if human_df is not None:
                    human_df.to_csv(f_human, index=False)
                    print(f"[Info] Human evaluation data saved to {f_human}")
                else:
                    print(f"[Warn] No human evaluation data found for {lang}")
                STATS.end(f"Emp - {lang} - Human Eval")
            
            STATS.end(f"Emp - {lang} - Total")
            print(f"[Success] {lang} dataset updated.")


def main():
    """
    Main execution function for Module 1: Data Processing.
    
    Runs both synthetic and empirical data processing.
    """
    print(f"\n{'='*40}")
    print(f"PMIScore Module 1: Data Processing")
    print(f"Output Base: {Config.BASE_DIR}")
    print(f"{'='*40}\n")
    
    # Process synthetic datasets
    syn_processor = SyntheticDataProcessor()
    syn_processor.run()
    
    # Process empirical datasets
    emp_processor = EmpiricalDataProcessor()
    emp_processor.run()
    
    print(f"\n[Module 1 Finished] All datasets generated and processed.")
    STATS.summary("execution_summary_module1.txt")


if __name__ == "__main__":
    main()
