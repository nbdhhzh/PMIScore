# -*- coding: utf-8 -*-
"""
PMIScore Module 2: Embedding Generation

This module handles:
1. Generation of embeddings for multiple LLM models using vLLM
2. MEEP direct scoring via text generation
3. Support for 6 models: Llama-3.2-3B, Phi-4-mini, Qwen3-0.6B/1.7B/4B/8B
4. Generation of 4 embedding types: context, response, prompt_meeep, prompt_pmi

Output: 
- embeddings/{synthetic,empirical}/{dataset}/{model}/: .npy files
- meep_scores/{synthetic,empirical}/{dataset}/{model}/: CSV files

Requirements: GPU with vLLM support
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from config import Config, ExecutionStats

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize execution stats
STATS = ExecutionStats()


def install_dependencies():
    """Install required dependencies for vLLM, GPU processing, and OpenAI SDK."""
    print("[Info] Installing vLLM and other dependencies...")
    os.system("pip install vllm openai")


def clear_gpu_memory():
    """
    Clear GPU memory to avoid OOM errors.
    
    This function attempts to clear GPU cache using PyTorch.
    """
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


class ModelProcessor:
    """
    Process embeddings and MEEP scores for a single model.
    
    This class handles:
    - vLLM initialization for embedding and generation tasks
    - Embedding generation for context, response, and prompted inputs
    - MEEP direct scoring via text generation
    """
    
    def __init__(self, model_path, force_overwrite=False):
        """
        Initialize ModelProcessor.
        
        Args:
            model_path: Path to the model (e.g., "Qwen/Qwen3-4B").
            force_overwrite: If True, regenerate embeddings even if they exist.
        """
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.force_overwrite = force_overwrite
        self.llm = None
    
    def _should_skip(self, npy_path, expected_count):
        """
        Check if embedding file should be skipped.
        
        Args:
            npy_path: Path to the .npy file.
            expected_count: Expected number of embeddings.
        
        Returns:
            True if file exists with correct count, False otherwise.
        """
        if self.force_overwrite:
            return False
        if not npy_path.exists():
            return False
        try:
            data = np.load(npy_path)
            if data.shape[0] == expected_count:
                print(f"[Skip] {npy_path.name} exists with correct count ({expected_count})")
                return True
            else:
                print(f"[Regenerate] {npy_path.name}: expected {expected_count}, got {data.shape[0]}")
                return False
        except Exception as e:
            print(f"[Regenerate] {npy_path.name}: load error - {e}")
            return False
    
    def _init_vllm(self, task="generate"):
        """
        Initialize vLLM model.
        
        Args:
            task: Type of task - "embed" for embeddings, "generate" for text generation.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        if self.llm is not None:
            del self.llm
            clear_gpu_memory()
        
        try:
            from vllm import LLM
        except ImportError:
            print("[Error] vLLM not installed. Run: pip install vllm")
            return False
        
        stat_name = f"vLLM Init - {self.model_name} ({task})"
        STATS.start(stat_name)
        
        print(f"[{self.model_name}] Initializing vLLM for task: {task}...")
        try:
            common_params = {
                "model": self.model_path,
                "trust_remote_code": True,
                "dtype": "auto",
                "enforce_eager": True,
                "gpu_memory_utilization": Config.GPU_MEMORY_UTILIZATION,
                "max_model_len": Config.MAX_MODEL_LEN
            }
            
            if task == "embed":
                common_params["task"] = "embed"
            
            self.llm = LLM(**common_params)
            STATS.end(stat_name)
            return True
        except Exception as e:
            STATS.end(stat_name)
            print(f"[Error] Initializing vLLM: {e}")
            return False
    
    def _get_dataset_files(self):
        """
        Get list of dataset files to process.
        
        Returns:
            List of dictionaries containing dataset information:
            - type: 'synthetic' or 'empirical'
            - dataset_name: Name of the dataset
            - split: 'train', 'val', 'test', or 'human'
            - path: Full path to the CSV file
        """
        files = []
        
        # 1. Synthetic datasets (Structure: datasets/synthetic/{case_name})
        for case in Config.SYNTHETIC_CASES:
            base = Config.DATA_DIR / "synthetic" / case
            if not base.exists():
                continue
            
            # Priority: read files with negatives first
            for split in ["train", "val", "test"]:
                fname = f"{split}_with_negatives.csv"
                if (base / fname).exists():
                    files.append({
                        "type": "synthetic",
                        "dataset_name": case,
                        "split": split,
                        "path": base / fname
                    })
                elif (base / f"{split}.csv").exists():
                    files.append({
                        "type": "synthetic",
                        "dataset_name": case,
                        "split": split,
                        "path": base / f"{split}.csv"
                    })
        
        # 2. Empirical datasets (Structure: datasets/empirical/{lang_code})
        for lang in Config.EMPIRICAL_LANGS:
            base = Config.DATA_DIR / "empirical" / lang
            if not base.exists():
                continue
            
            # Train/Val/Test splits
            for split in ["train", "val", "test"]:
                fname = f"{split}_with_negatives.csv"
                if (base / fname).exists():
                    files.append({
                        "type": "empirical",
                        "dataset_name": lang,
                        "split": split,
                        "path": base / fname
                    })
            
            # Human evaluation split
            if (base / "human.csv").exists():
                files.append({
                    "type": "empirical",
                    "dataset_name": lang,
                    "split": "human",
                    "path": base / "human.csv"
                })
        
        return files
    
    def run_embeddings(self):
        """
        Generate embeddings for all datasets.
        
        Generates 4 types of embeddings per sample:
        1. Context embeddings (raw)
        2. Response embeddings (raw)
        3. Prompted embeddings (MEEP template)
        4. Prompted embeddings (PMI template)
        
        Output directory: embeddings/{type}/{dataset_name}/{model_name}/
        """
        if not self._init_vllm(task="embed"):
            return
        
        tasks = self._get_dataset_files()
        
        for task in tasks:
            ds_type = task['type']
            ds_name = task['dataset_name']
            split_name = task['split']
            file_path = task['path']
            
            # Build output path: maintain same hierarchy as input
            out_dir = Config.EMBEDDING_DIR / ds_type / ds_name / self.model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output file names
            out_ctx_npy = out_dir / f"{split_name}_context_embeddings.npy"
            out_rsp_npy = out_dir / f"{split_name}_response_embeddings.npy"
            out_pmt_npy = out_dir / f"{split_name}_prompt_embeddings.npy"
            out_pmt_pmi_npy = out_dir / f"{split_name}_prompt_pmi_embeddings.npy"
            out_meta = out_dir / f"{split_name}_meta.csv"
            
            op_name = f"{self.model_name} - Embed - {ds_type}/{ds_name}/{split_name}"
            STATS.start(op_name)
            print(f"Embedding {ds_type}/{ds_name} - {split_name} ...")
            df = pd.read_csv(file_path)
            expected_count = len(df)
            
            # Check if all files should be skipped (smart skip logic)
            all_skipped = (
                self._should_skip(out_ctx_npy, expected_count) and
                self._should_skip(out_rsp_npy, expected_count) and
                self._should_skip(out_pmt_npy, expected_count) and
                self._should_skip(out_pmt_pmi_npy, expected_count) and
                out_meta.exists()
            )
            if all_skipped:
                print(f"[Skip] All embeddings valid for {ds_type}/{ds_name}/{split_name}")
                STATS.end(op_name)
                continue
            
            # Prepare text lists
            contexts = []
            responses = []
            prompts_meep = []
            prompts_pmi = []
            
            for _, row in df.iterrows():
                # Use paraphrased text if available
                c = str(row.get('context_paraphrased', row.get('context', '')))
                r = str(row.get('response_paraphrased', row.get('response', '')))
                
                contexts.append(c)
                responses.append(r)
                # Generate prompted text
                prompts_meep.append(Config.MEEP_PROMPT_TEMPLATE.format(context=c, response=r))
                prompts_pmi.append(Config.PMI_PROMPT_TEMPLATE.format(context=c, response=r))
            
            try:
                # 1. Context Embeddings (Raw)
                if not self._should_skip(out_ctx_npy, expected_count):
                    sub_op_name = f"{op_name} - Context"
                    STATS.start(sub_op_name)
                    print("  -> Embedding Contexts...")
                    ctx_outputs = self.llm.embed(contexts)
                    ctx_embeddings = np.array([o.outputs.embedding for o in ctx_outputs])
                    np.save(out_ctx_npy, ctx_embeddings.astype(np.float32))
                    STATS.end(sub_op_name)
                
                # 2. Response Embeddings (Raw)
                if not self._should_skip(out_rsp_npy, expected_count):
                    sub_op_name = f"{op_name} - Response"
                    STATS.start(sub_op_name)
                    print("  -> Embedding Responses...")
                    rsp_outputs = self.llm.embed(responses)
                    rsp_embeddings = np.array([o.outputs.embedding for o in rsp_outputs])
                    np.save(out_rsp_npy, rsp_embeddings.astype(np.float32))
                    STATS.end(sub_op_name)
                
                # 3. Prompt Embeddings (MEEP Template)
                if not self._should_skip(out_pmt_npy, expected_count):
                    sub_op_name = f"{op_name} - Prompt_MEEP"
                    STATS.start(sub_op_name)
                    print("  -> Embedding Prompts_MEEP...")
                    pmt_outputs = self.llm.embed(prompts_meep)
                    pmt_embeddings = np.array([o.outputs.embedding for o in pmt_outputs])
                    np.save(out_pmt_npy, pmt_embeddings.astype(np.float32))
                    STATS.end(sub_op_name)
                
                # 4. Prompt Embeddings (PMI Template)
                if not self._should_skip(out_pmt_pmi_npy, expected_count):
                    sub_op_name = f"{op_name} - Prompt_PMI"
                    STATS.start(sub_op_name)
                    print("  -> Embedding Prompts_PMI...")
                    pmt_pmi_outputs = self.llm.embed(prompts_pmi)
                    pmt_pmi_embeddings = np.array([o.outputs.embedding for o in pmt_pmi_outputs])
                    np.save(out_pmt_pmi_npy, pmt_pmi_embeddings.astype(np.float32))
                    STATS.end(sub_op_name)
                
            except AttributeError:
                print(f"[Error] Model {self.model_name} does not support embedding generation via vLLM.")
                STATS.end(op_name)
                continue
            except Exception as e:
                print(f"[Error] Embedding generation failed: {e}")
                STATS.end(op_name)
                continue
            
            # Save metadata
            cols_to_exclude = [
                'context', 'response', 'context_paraphrased', 'response_paraphrased',
                'joint_file', 'px_file', 'py_file'
            ]
            meta_cols = [c for c in df.columns if c not in cols_to_exclude]
            df[meta_cols].to_csv(out_meta, index=False)
            
            print(f"Saved 4 embedding sets for {ds_name}/{split_name}")
            STATS.end(op_name)
        
        # Release GPU memory
        del self.llm
        self.llm = None
        clear_gpu_memory()
    
    def run_meep_scoring(self):
        """
        Generate MEEP direct scores via text generation.
        
        Uses the MEEP prompt to generate engagement scores (N=5 samples per prompt).
        Computes mean and standard deviation of the scores.
        
        Output directory: meep_scores/{type}/{dataset_name}/{model_name}/
        """
        if not self._init_vllm(task="generate"):
            return
        
        from vllm import SamplingParams
        
        # Set sampling parameters: N=5 samples with temperature for diversity
        sampling_params = SamplingParams(
            temperature=Config.MEEP_TEMPERATURE,
            n=Config.MEEP_NUM_SAMPLES,
            max_tokens=Config.MEEP_MAX_TOKENS
        )
        
        tasks = self._get_dataset_files()
        
        for task in tasks:
            ds_type = task['type']
            ds_name = task['dataset_name']
            split_name = task['split']
            file_path = task['path']
            
            # Only process test and human splits for MEEP scoring
            if split_name not in ["test", "human"]:
                continue
            
            # Build output path
            out_dir = Config.SCORE_DIR / ds_type / ds_name / self.model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{split_name}_scores.csv"
            
            if out_csv.exists():
                print(f"[Skip] Scores already exist: {out_csv}")
                continue
            
            op_name = f"{self.model_name} - Score - {ds_type}/{ds_name}/{split_name}"
            STATS.start(op_name)
            
            print(f"Scoring {ds_type}/{ds_name} - {split_name} (N={Config.MEEP_NUM_SAMPLES}) ...")
            df = pd.read_csv(file_path)
            
            prompts = []
            for _, row in df.iterrows():
                ctx = str(row.get('context_paraphrased', row.get('context', '')))
                rsp = str(row.get('response_paraphrased', row.get('response', '')))
                prompts.append(Config.MEEP_PROMPT_TEMPLATE.format(context=ctx, response=rsp))
            
            try:
                # Batch generation: vLLM generates N outputs for each prompt
                outputs = self.llm.generate(prompts, sampling_params)
                
                data_rows = []
                
                # Iterate over each prompt's results (RequestOutput)
                for idx, request_output in enumerate(outputs):
                    raw_texts_list = []
                    scores_list = []
                    
                    # Iterate over the N generated results (CompletionOutput)
                    for completion in request_output.outputs:
                        text = completion.text
                        score = extract_score(text)
                        
                        raw_texts_list.append(text)
                        scores_list.append(score)
                    
                    # Compute statistics (filter out failed parsing)
                    valid_scores = [s for s in scores_list if s is not None]
                    
                    if valid_scores:
                        mean_score = np.mean(valid_scores)
                        std_dev = np.std(valid_scores)
                    else:
                        mean_score = None
                        std_dev = None
                    
                    data_rows.append({
                        'idx': idx,
                        'score_mean': mean_score,
                        'score_std': std_dev,
                        'scores_raw': json.dumps(scores_list),
                        'llm_outputs_raw': json.dumps(raw_texts_list)
                    })
                
                out_df = pd.DataFrame(data_rows)
                out_df.to_csv(out_csv, index=False)
                print(f"Saved aggregated scores (N={Config.MEEP_NUM_SAMPLES}) to {out_csv}")
                STATS.end(op_name)
            
            except Exception as e:
                print(f"[Error] Scoring failed for {ds_name}/{split_name}: {e}")
                import traceback
                traceback.print_exc()
                STATS.end(op_name)
        
        del self.llm
        self.llm = None
        clear_gpu_memory()


def extract_score(text):
    """
    Extract a numeric score from generated text.
    
    Args:
        text: Generated text from the model.
    
    Returns:
        Extracted score as float, or None if extraction fails.
    """
    import re
    
    # Try to extract a number from the text
    # Look for patterns like "7", "7.5", "score: 8", etc.
    patterns = [
        r'\b(\d+\.?\d*)\b',  # Any number
        r'(\d+)\s*/\s*10',  # Score out of 10
        r'score[:\s]*(-?\d+\.?\d*)',  # "score: 7.5"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                score = float(matches[0])
                # Clip score to reasonable range [0, 10]
                score = max(0, min(10, score))
                return score
            except ValueError:
                continue
    
    return None


class OpenRouterEmbeddingProcessor:
    """
    Process embeddings using OpenRouter API for embedding-only models.
    
    This class handles:
    - Async batch API calls to OpenRouter embedding endpoint
    - Parallel processing with rate limiting
    - Smart skip logic based on file existence and data count
    """
    
    def __init__(self, model_id, force_overwrite=False):
        """
        Initialize OpenRouterEmbeddingProcessor.
        
        Args:
            model_id: OpenRouter model ID (e.g., "qwen/qwen3-embedding-8b").
            force_overwrite: If True, regenerate embeddings even if they exist.
        """
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.force_overwrite = force_overwrite
    
    def _should_skip(self, npy_path, expected_count):
        """
        Check if embedding file should be skipped.
        
        Args:
            npy_path: Path to the .npy file.
            expected_count: Expected number of embeddings.
        
        Returns:
            True if file exists with correct count, False otherwise.
        """
        if self.force_overwrite:
            return False
        if not npy_path.exists():
            return False
        try:
            data = np.load(npy_path)
            if data.shape[0] == expected_count:
                print(f"[Skip] {npy_path.name} exists with correct count ({expected_count})")
                return True
            else:
                print(f"[Regenerate] {npy_path.name}: expected {expected_count}, got {data.shape[0]}")
                return False
        except Exception as e:
            print(f"[Regenerate] {npy_path.name}: load error - {e}")
            return False
    
    def _get_dataset_files(self):
        """
        Get list of dataset files to process.
        
        Returns:
            List of dictionaries containing dataset information.
        """
        files = []
        
        for case in Config.SYNTHETIC_CASES:
            base = Config.DATA_DIR / "synthetic" / case
            if not base.exists():
                continue
            for split in ["train", "val", "test"]:
                fname = f"{split}_with_negatives.csv"
                if (base / fname).exists():
                    files.append({
                        "type": "synthetic",
                        "dataset_name": case,
                        "split": split,
                        "path": base / fname
                    })
                elif (base / f"{split}.csv").exists():
                    files.append({
                        "type": "synthetic",
                        "dataset_name": case,
                        "split": split,
                        "path": base / f"{split}.csv"
                    })
        
        for lang in Config.EMPIRICAL_LANGS:
            base = Config.DATA_DIR / "empirical" / lang
            if not base.exists():
                continue
            for split in ["train", "val", "test"]:
                fname = f"{split}_with_negatives.csv"
                if (base / fname).exists():
                    files.append({
                        "type": "empirical",
                        "dataset_name": lang,
                        "split": split,
                        "path": base / fname
                    })
            if (base / "human.csv").exists():
                files.append({
                    "type": "empirical",
                    "dataset_name": lang,
                    "split": "human",
                    "path": base / "human.csv"
                })
        
        return files
    
    def _init_client(self):
        """Initialize OpenAI client for OpenRouter."""
        if OpenAI is None:
            raise ImportError("openai is required. Run: pip install openai")
        
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=Config.API_KEY,
        )
    
    def _embed_batch(self, client, texts, batch_idx):
        """
        Send a batch of texts to OpenRouter embedding API.
        
        Args:
            client: OpenAI client instance.
            texts: List of texts to embed.
            batch_idx: Batch index for error reporting.
        
        Returns:
            Tuple of (batch_idx, list of embedding vectors) or (batch_idx, None) on error.
        """
        try:
            response = client.embeddings.create(
                model=self.model_id,
                input=texts,
                encoding_format="float",
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]
            return (batch_idx, embeddings)
        except Exception as e:
            print(f"[Error] Batch {batch_idx} failed: {e}")
            return (batch_idx, None)
    
    def embed_texts(self, texts):
        """
        Embed all texts using parallel API calls with ThreadPoolExecutor.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            numpy.ndarray: Embedding matrix.
        """
        client = self._init_client()
        
        batch_size = Config.OPENROUTER_BATCH_SIZE
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        results = [None] * len(batches)
        
        with ThreadPoolExecutor(max_workers=Config.OPENROUTER_MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(self._embed_batch, client, batch, i): i 
                for i, batch in enumerate(batches)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="API Batches"):
                batch_idx, embeddings = future.result()
                results[batch_idx] = embeddings
        
        embed_dim = None
        for r in results:
            if r is not None and len(r) > 0:
                embed_dim = len(r[0])
                break
        
        if embed_dim is None:
            raise Exception("All embedding batches failed")
        
        all_embeddings = []
        for i, batch in enumerate(batches):
            batch_result = results[i]
            if batch_result is not None:
                all_embeddings.extend(batch_result)
            else:
                all_embeddings.extend([[0.0] * embed_dim] * len(batch))
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def run_embeddings(self):
        """
        Generate embeddings for all datasets using OpenRouter API.
        
        Generates 4 types of embeddings per sample:
        1. Context embeddings (raw)
        2. Response embeddings (raw)
        3. Prompted embeddings (MEEP template)
        4. Prompted embeddings (PMI template)
        """
        if Config.API_KEY is None:
            print("[Error] OPENROUTER_API_KEY not set. Skipping OpenRouter embeddings.")
            return
        
        tasks = self._get_dataset_files()
        
        for task in tasks:
            ds_type = task['type']
            ds_name = task['dataset_name']
            split_name = task['split']
            file_path = task['path']
            
            out_dir = Config.EMBEDDING_DIR / ds_type / ds_name / self.model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_ctx_npy = out_dir / f"{split_name}_context_embeddings.npy"
            out_rsp_npy = out_dir / f"{split_name}_response_embeddings.npy"
            out_pmt_npy = out_dir / f"{split_name}_prompt_embeddings.npy"
            out_pmt_pmi_npy = out_dir / f"{split_name}_prompt_pmi_embeddings.npy"
            out_meta = out_dir / f"{split_name}_meta.csv"
            
            op_name = f"{self.model_name} - Embed - {ds_type}/{ds_name}/{split_name}"
            STATS.start(op_name)
            print(f"[OpenRouter] Embedding {ds_type}/{ds_name} - {split_name} ...")
            
            df = pd.read_csv(file_path)
            expected_count = len(df)
            
            all_skipped = (
                self._should_skip(out_ctx_npy, expected_count) and
                self._should_skip(out_rsp_npy, expected_count) and
                self._should_skip(out_pmt_npy, expected_count) and
                self._should_skip(out_pmt_pmi_npy, expected_count) and
                out_meta.exists()
            )
            if all_skipped:
                print(f"[Skip] All embeddings valid for {ds_type}/{ds_name}/{split_name}")
                STATS.end(op_name)
                continue
            
            contexts = []
            responses = []
            prompts_meep = []
            prompts_pmi = []
            
            for _, row in df.iterrows():
                c = str(row.get('context_paraphrased', row.get('context', '')))
                r = str(row.get('response_paraphrased', row.get('response', '')))
                contexts.append(c)
                responses.append(r)
                prompts_meep.append(Config.MEEP_PROMPT_TEMPLATE.format(context=c, response=r))
                prompts_pmi.append(Config.PMI_PROMPT_TEMPLATE.format(context=c, response=r))
            
            try:
                if not self._should_skip(out_ctx_npy, expected_count):
                    print("  -> Embedding Contexts...")
                    ctx_embeddings = self.embed_texts(contexts)
                    np.save(out_ctx_npy, ctx_embeddings)
                
                if not self._should_skip(out_rsp_npy, expected_count):
                    print("  -> Embedding Responses...")
                    rsp_embeddings = self.embed_texts(responses)
                    np.save(out_rsp_npy, rsp_embeddings)
                
                if not self._should_skip(out_pmt_npy, expected_count):
                    print("  -> Embedding Prompts_MEEP...")
                    pmt_embeddings = self.embed_texts(prompts_meep)
                    np.save(out_pmt_npy, pmt_embeddings)
                
                if not self._should_skip(out_pmt_pmi_npy, expected_count):
                    print("  -> Embedding Prompts_PMI...")
                    pmt_pmi_embeddings = self.embed_texts(prompts_pmi)
                    np.save(out_pmt_pmi_npy, pmt_pmi_embeddings)
                
            except Exception as e:
                print(f"[Error] OpenRouter embedding failed: {e}")
                import traceback
                traceback.print_exc()
                STATS.end(op_name)
                continue
            
            cols_to_exclude = [
                'context', 'response', 'context_paraphrased', 'response_paraphrased',
                'joint_file', 'px_file', 'py_file'
            ]
            meta_cols = [c for c in df.columns if c not in cols_to_exclude]
            df[meta_cols].to_csv(out_meta, index=False)
            
            print(f"Saved 4 embedding sets for {ds_name}/{split_name}")
            STATS.end(op_name)


def main():
    """
    Main execution function for Module 2: Embedding Generation.
    
    Runs embedding generation and MEEP scoring for all configured models.
    """
    parser = argparse.ArgumentParser(description="PMIScore Module 2: Embedding Generation")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Force regenerate embeddings even if they exist with correct count")
    parser.add_argument("--openrouter-only", action="store_true",
                        help="Only run OpenRouter embedding models (skip vLLM models)")
    parser.add_argument("--vllm-only", action="store_true",
                        help="Only run vLLM models (skip OpenRouter models)")
    args = parser.parse_args()
    
    install_dependencies()
    
    print(f"\n{'='*40}")
    print(f"PMIScore Module 2: Embedding Generation")
    print(f"Output Base: {Config.BASE_DIR}")
    print(f"Structure:   {{BASE}}/{{type}}/{{dataset}}/{{model}}/...")
    print(f"vLLM Models: {len(Config.MODELS)}")
    print(f"OpenRouter:  {len(Config.OPENROUTER_EMBEDDING_MODELS)}")
    print(f"Force Overwrite: {args.force_overwrite}")
    print(f"{'='*40}\n")
    
    if not args.vllm_only:
        for model_id in Config.OPENROUTER_EMBEDDING_MODELS:
            print(f"\n>>> Processing OpenRouter Model: {model_id} <<<")
            processor = OpenRouterEmbeddingProcessor(model_id, args.force_overwrite)
            try:
                processor.run_embeddings()
            except Exception as e:
                print(f"[Error] OpenRouter embedding for {model_id}: {e}")
    
    if not args.openrouter_only:
        for model_path in Config.MODELS:
            print(f"\n>>> Processing vLLM Model: {model_path} <<<")
            processor = ModelProcessor(model_path, args.force_overwrite)
            
            try:
                processor.run_embeddings()
            except Exception as e:
                print(f"[Error] Embedding for {model_path}: {e}")
                clear_gpu_memory()
            
            try:
                processor.run_meep_scoring()
            except Exception as e:
                print(f"[Error] Scoring for {model_path}: {e}")
                clear_gpu_memory()
    
    print("\n[Module 2 Finished] All embeddings and scores generated.")
    STATS.summary("execution_summary_module2.txt")


if __name__ == "__main__":
    main()
