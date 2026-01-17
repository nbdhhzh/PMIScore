# -*- coding: utf-8 -*-
"""
PMIScore Configuration Module

Centralized configuration for all PMIScore pipeline modules.
This module provides shared configuration parameters and utility classes
to ensure consistency across data processing, embedding generation,
training/evaluation, and visualization modules.
"""

import os
import time
import io
import contextlib
from pathlib import Path

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

class Config:
    """
    Main configuration class for PMIScore pipeline.
    
    All paths and parameters are defined here for consistency
    across all modules.
    """
    
    # ==========================================
    # PATH CONFIGURATION
    # ==========================================
    
    # Base directory for all outputs
    BASE_DIR = Path("./")
    
    # Data directories
    DATA_DIR = BASE_DIR / "datasets"
    SYNTHETIC_DIR = DATA_DIR / "synthetic"
    EMPIRICAL_DIR = DATA_DIR / "empirical"
    
    # Output directories
    EMBEDDING_DIR = BASE_DIR / "embeddings"
    RESULT_DIR = BASE_DIR / "results"
    SCORE_DIR = BASE_DIR / "meep_scores"
    OUTPUT_DIR = BASE_DIR / "analysis_report"
    
    # ==========================================
    # CORE TRAINING CONFIGURATION
    # ==========================================
    
    # Negative sampling
    TRAIN_NEG_RATIO = 15  # Number of negative samples per positive
    NEG_IN_DIALOGUE_PROB = 0.1  # Probability of negative sample in dialogue
    NUM_ROUNDS = 5  # Number of training rounds
    NEG_SAMPLES_USED = 3  # Number of negatives used per round (out of 15)
    GROUP_SIZE = 1 + (NUM_ROUNDS * NEG_SAMPLES_USED)  # 1 pos + (5 * 3) neg = 16
    
    # Random seed for reproducibility
    SEED = 42
    
    # ==========================================
    # SYNTHETIC DATASET CONFIGURATION
    # ==========================================
    
    SYN_SIZE = 20  # Size of joint matrix (20x20)
    SYN_SAMPLES = 5000  # Total number of samples per dataset
    SYN_SPLIT = (0.6, 0.2, 0.2)  # Train/val/test split ratios
    SYNTHETIC_CASES = ["diagonal", "independent", "block"]
    
    # ==========================================
    # EMPIRICAL DATASET CONFIGURATION
    # ==========================================
    
    # DSTC11 data download
    DSTC_DOWNLOAD_DIR = "DSTC_11_Track_4"
    DSTC_ZIP_URL = "https://huggingface.co/datasets/mario-rc/dstc11.t4/resolve/main/DSTC_11_Track_4.zip"
    
    # Empirical dataset sizes
    EMP_SAMPLES = 5000
    EMP_SPLIT = (0.6, 0.2, 0.2)
    EMPIRICAL_LANGS = ["en", "zh"]
    
    # Human evaluation files for each language
    HUMAN_EVAL_FILES = {
        "en": ["en/fed-turn/fed-turn_eval.json"],
        "zh": ["zh/KdConv/KdConv-turn_eval.json", "zh/LCCC/LCCC-turn_eval.json"]
    }
    
    # Length and turn limits for context
    MAX_CONTEXT_CHARS = 30000
    MAX_CONTEXT_TURNS = 10
    
    # ==========================================
    # EMBEDDING GENERATION CONFIGURATION
    # ==========================================
    
    # Prompt templates
    MEEP_PROMPT_TEMPLATE = (
        "Context: {context}\n"
        "Response: {response}\n"
        "On a scale of 1-10, how engaging is this response? Output ONLY a number."
    )
    
    PMI_PROMPT_TEMPLATE = (
        "Context: {context}\n"
        "Response: {response}\n"
        "Estimate the pointwise mutual information between this context and response. "
        "Output a score."
    )
    
    # Model list for embedding generation
    MODELS = [
        "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
    ]
    
    # vLLM configuration
    MAX_MODEL_LEN = 8192
    GPU_MEMORY_UTILIZATION = 0.95
    
    # MEEP scoring configuration
    MEEP_NUM_SAMPLES = 5  # Number of samples per prompt
    MEEP_TEMPERATURE = 0.7  # Temperature for diversity
    MEEP_MAX_TOKENS = 16
    
    # ==========================================
    # TRAINING CONFIGURATION
    # ==========================================
    
    # Neural network hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 256
    LR = 1e-3
    
    # KDE configuration
    KDE_PCA_DIM = 128
    KDE_BANDWIDTH = 'auto'
    KDE_INFER_BATCH_SIZE = 2048
    
    # Neural network configurations (Name, Loss, InputType)
    NN_CONFIGS = [
        ("PMI_Pair", "pmi_nce", "pair"),
        ("MINE_Pair", "mine", "pair"),
        ("InfoNCE_Pair", "infonce", "pair"),
        ("PMI_Prompt", "pmi_nce", "single"),
        ("MINE_Prompt", "mine", "single"),
        ("InfoNCE_Prompt", "infonce", "single"),
    ]
    
    # KDE configurations (Name, Mode)
    KDE_CONFIGS = [
        ("KDE_Pair", "pair_diff"),
        ("KDE_Prompt", "single_diff"),
    ]
    
    # ==========================================
    # VISUALIZATION CONFIGURATION
    # ==========================================
    
    # Color mapping for methods
    COLORS = {
        "PMIScore": "#1f77b4",
        "MINE": "#ff7f0e",
        "InfoNCE": "#2ca02c",
        "KDE": "#d62728",
        "MEEP": "#9467bd",
        "Direct_MEEP": "#9467bd"
    }
    
    # Method name mapping (internal -> display)
    METHOD_MAP = {
        "PMI_Prompt": "PMIScore",
        "MINE_Prompt": "MINE",
        "InfoNCE_Prompt": "InfoNCE",
        "KDE_Prompt": "KDE",
        "Direct_MEEP": "MEEP"
    }
    
    # Target methods for analysis
    TARGET_METHODS = ["PMIScore", "MINE", "InfoNCE", "KDE", "MEEP"]
    TABLE_METHODS = ["PMIScore", "MINE", "InfoNCE", "KDE", "MEEP"]
    
    # Plotting settings
    FIGURE_DPI = 300
    FIGURE_FORMATS = ["png", "pdf"]
    
    # ==========================================
    # API CONFIGURATION (Optional)
    # ==========================================
    
    # OpenRouter API key for paraphrasing (optional)
    # Set via environment variable: export OPENROUTER_API_KEY="your_key"
    API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MAX_WORKERS = 20
    
    # Paraphrasing styles
    PARAPHRASE_STYLES = [
        "formal and professional",
        "casual and conversational",
        "warm and friendly",
        "polite and respectful"
    ]
    
    @staticmethod
    def init_directories():
        """
        Initialize all required directories.
        
        Creates the directory structure if it doesn't exist.
        """
        directories = [
            Config.SYNTHETIC_DIR,
            Config.EMPIRICAL_DIR,
            Config.EMBEDDING_DIR,
            Config.RESULT_DIR,
            Config.SCORE_DIR,
            Config.OUTPUT_DIR
        ]
        
        for p in directories:
            p.mkdir(parents=True, exist_ok=True)
        
        print(f"[Init] Directories ready at {Config.BASE_DIR}")
    
    @staticmethod
    def set_seed(seed=None):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed. If None, uses Config.SEED.
        """
        if seed is None:
            seed = Config.SEED
        
        try:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
        except ImportError:
            pass


# ==========================================
# UTILITY CLASSES
# ==========================================

class ExecutionStats:
    """
    Utility class for tracking execution time of different steps.
    
    Tracks start and end times for named tasks and provides
    a summary report at the end.
    """
    
    def __init__(self):
        """Initialize ExecutionStats with empty stats dictionary."""
        self.stats = {}
        self.current_start = None
    
    def start(self, name):
        """
        Start timing a named task.
        
        Args:
            name: Name of the task to time.
        
        Returns:
            The task name for chaining.
        """
        self.current_start = time.time()
        print(f"\n[Step Start] {name}...")
        return name
    
    def end(self, name):
        """
        End timing a named task.
        
        Args:
            name: Name of the task to end timing for.
        """
        if self.current_start is None:
            return
        
        duration = time.time() - self.current_start
        self.stats[name] = duration
        print(f"[Step Done] {name} - Taken: {duration:.2f}s")
        self.current_start = None
    
    def summary(self, filename="execution_summary.txt"):
        """
        Print and save execution summary.
        
        Args:
            filename: Name of the file to save the summary to.
        """
        save_path = Config.BASE_DIR / filename
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            print(f"\n{'='*20} Execution Summary {'='*20}")
            print(f"{'Task Name':<40} | {'Duration (s)':<15}")
            print("-" * 60)
            total = 0
            for k, v in self.stats.items():
                print(f"{k:<40} | {v:.2f}")
                total += v
            print("-" * 60)
            print(f"{'Total Time':<40} | {total:.2f}s")
            print("=" * 60)
            summary_output = buf.getvalue()
        
        # Print to console
        print(summary_output)
        
        # Save to file
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(summary_output)
            print(f"[Info] Execution summary saved to {save_path}")
        except Exception as e:
            print(f"[Warn] Could not save execution summary to file {save_path}: {e}")


# ==========================================
# INITIALIZATION
# ==========================================

# Set seed and initialize directories when module is imported
Config.set_seed()
Config.init_directories()
