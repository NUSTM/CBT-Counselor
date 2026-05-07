#!/usr/bin/env python3
"""
Single-GPU SFT model validation script.
Loads an FSDP2-trained merged model checkpoint for single-GPU inference.
"""

import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from CBT_Counselor.evaluation_metric import MetricsCalculator
from verl.utils import hf_tokenizer
from verl.utils.dataset.sft_counselor_dataset import SFTDataset
from verl.utils.fs import copy_to_local


os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleGPUValidator:

    def __init__(self, config, validation_time):
        self.config = config
        self.checkpoint_path = config.model.checkpoint_path
        self.device = config.rollout.device

        if self.config.cbt_guided:
            self.model_name = self.config.model.partial_pretrain.split("/")[-1] + "_cbt_guided"
        else:
            self.model_name = self.config.model.partial_pretrain.split("/")[-1] + "_no_cbt_guided"

        self.validation_time = validation_time

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
            print(f"Using device: {self.device}")
        else:
            self.device = "cpu"
            print("CUDA not available, using CPU")

        self._load_tokenizer()
        self._load_datasets()
        self._load_model()

        self.metrics_calculator = MetricsCalculator()

        print(f"Single-GPU SFT Validator initialized successfully")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Validation samples: {len(self.val_dataset)}")

    def _load_tokenizer(self):
        local_model_path = copy_to_local(src=self.config.model.checkpoint_path, verbose=True)
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        print(f"Tokenizer loaded from: {local_model_path}")

    def _load_datasets(self):
        val_files = self.config.data.val_files

        print(f"Loading validation dataset: {val_files}")

        self.val_dataset = SFTDataset(
            parquet_files=val_files,
            tokenizer=self.tokenizer,
            config=self.config,
            mode="eval"
        )

        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        print(f"Validation dataset loaded: {len(self.val_dataset)} samples")
        print(f"Validation batches: {len(self.val_dataloader)}")

    def _load_model(self):
        """Load the merged LoRA adapter model directly."""

        local_model_path = copy_to_local(src=self.config.model.checkpoint_path, verbose=True)

        trust_remote_code = self.config.model.trust_remote_code
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config

        print(f"Model config: model_type={getattr(config, 'model_type', 'unknown')}, "
              f"architectures={getattr(config, 'architectures', 'unknown')}")

        if hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = max(
                config.max_position_embeddings,
                self.config.data.max_length
            )

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "bf16")
        if torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"Loading model: {self.model_name}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
            device_map=None,
        )

        print(f"Model loaded: {self.model_name}")

        self.model = self.model.to(self.device)
        self.model.eval()


    def validate(self) -> Optional[Dict]:
        print("Starting single-GPU inference validation")

        results_file = self._setup_results_directory()
        self._run_validation_loop(results_file)

        if results_file:
            self._finalize_validation_results(results_file)

    def _setup_results_directory(self) -> Path:
        save_dir = Path(self.config.rollout.sft_validation_files_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        eval_set_name = self.config.data.val_files.split("/")[-1].split("_")[0]
        experi_setting = self.config.model.checkpoint_path.split("/")[-2]
        results_file = save_dir / f"{self.model_name}_{experi_setting}_{eval_set_name}_{self.validation_time}_predict.json"

        if results_file.exists():
            print(f"Clearing existing results file: {results_file}")
            results_file.unlink()

        results_file.touch()

        print(f"Results will be saved to: {results_file}")

        return results_file


    def _run_validation_loop(self, results_file: Optional[Path]) -> None:
        with torch.no_grad():
            for batch_idx, val_data in enumerate(tqdm(self.val_dataloader, desc="Validation")):

                sample_indices = val_data["idx"]

                batch_references = []
                batch_last_speaker_utterance_for_EPITOME = []

                for sample_idx in sample_indices:
                    sample_idx = sample_idx.item()
                    ref = self.val_dataset._get_reference_response(sample_idx)
                    last_utterance = self.val_dataset._get_last_spkeaker_utterance(sample_idx)
                    batch_references.append(ref)
                    batch_last_speaker_utterance_for_EPITOME.append(last_utterance)

                batch_predictions = self._process_batch(val_data)

                if results_file:
                    with open(results_file, 'a', encoding='utf-8') as f:
                        for pred, ref, last_speaker_utterance in zip(batch_predictions, batch_references, batch_last_speaker_utterance_for_EPITOME):
                            sample_data = {
                                'prediction': pred,
                                'reference': ref,
                                'last_speaker_utterance': last_speaker_utterance
                            }
                            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')


    def _process_batch(self, val_data: Dict) -> List[str]:
        input_ids = val_data["input_ids"].to(self.device)
        attention_mask = val_data["attention_mask"].to(self.device)

        if self.config.rollout.repetition_penalty:
            generation_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_new_tokens': self.config.rollout.max_new_tokens,
                'do_sample': self.config.rollout.do_sample,
                'repetition_penalty': self.config.rollout.repetition_penalty,
                'temperature': self.config.rollout.temperature,
                'top_p': self.config.rollout.top_p,
                'top_k': self.config.rollout.top_k,
                'min_p': self.config.rollout.min_p,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'use_cache': True,
            }
        else:
            generation_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_new_tokens': self.config.rollout.max_new_tokens,
                'do_sample': self.config.rollout.do_sample,
                'temperature': self.config.rollout.temperature,
                'top_p': self.config.rollout.top_p,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'use_cache': True,
            }

        outputs = self.model.generate(**generation_kwargs)

        generated_texts = self._extract_generated_tokens(
            outputs, input_ids, attention_mask
        )
        return generated_texts

    def _extract_generated_tokens(self, full_outputs: torch.Tensor,
                                    original_input_ids: torch.Tensor,
                                    attention_mask: torch.Tensor) -> List[str]:
        """Extract newly generated tokens from full model output."""
        batch_size = full_outputs.shape[0]
        original_seq_len = original_input_ids.shape[1]

        generated_texts = []

        for i in range(batch_size):
            if full_outputs.shape[1] > original_seq_len:
                generated_tokens = full_outputs[i, original_seq_len:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""

            if SFTDataset.get_thinking_flag():
                generated_text = self._extract_content_from_thinking(generated_text)

            generated_texts.append(generated_text.strip())

        return generated_texts

    def _extract_content_from_thinking(self, text):
        """Strip <think>...</think> blocks from Qwen3 thinking-mode output."""
        pattern = r'<think>.*?</think>'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text.strip()


    def _finalize_validation_results(self, results_file: Path) -> Dict:
        print("Computing final metrics from saved results...")
        final_metrics = self._compute_metrics_from_file(results_file)

        eval_set_name = self.config.data.val_files.split("/")[-1].split("_")[0]
        experi_setting = self.config.model.checkpoint_path.split("/")[-2]
        results_path = Path(results_file)
        metrics_file = results_path.parent / f"{self.model_name}_{experi_setting}_{eval_set_name}_{self.validation_time}_metrics.json"

        if metrics_file.exists():
            print(f"Clearing existing metrics file: {metrics_file}")
            metrics_file.unlink()

        metrics_file.touch()

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
            print(f"Final metrics saved to: {metrics_file}")


    def _compute_metrics_from_file(self, results_file: Path) -> Dict:
        all_references = []
        all_predictions = []
        all_last_speaker_utterances = []

        print(f"Reading results from {results_file}...")

        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample_data = json.loads(line)
                    all_references.append(sample_data["reference"])
                    all_predictions.append(sample_data["prediction"])
                    all_last_speaker_utterances.append(sample_data["last_speaker_utterance"])

        cbt_guided = self.config.cbt_guided

        metrics = self.metrics_calculator._compute_evaluation_metrics(all_references, all_predictions, all_last_speaker_utterances, cbt_guided)

        return metrics


    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        torch.cuda.empty_cache()



def main():
    import argparse
    import os

    start_time = time.time()

    start_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Validation started: {start_datetime}")

    parser = argparse.ArgumentParser(description="Single-GPU SFT Model Validation")
    parser.add_argument("--config_path", type=str, default="/path/to/CBT_Counselor/counselor_configs/sft_val_config.yaml",
                        help="Path to configuration file")

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    config = OmegaConf.load(args.config_path)

    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    base_path = config.model.partial_pretrain
    lora_path = config.model.lora_path
    merged_path = config.model.checkpoint_path

    if not os.path.exists(merged_path):
        print(f"Merging LoRA adapter into base model: {merged_path}")

        os.mkdir(merged_path)

        base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        model = PeftModel.from_pretrained(base_model, lora_path)
        merged = model.merge_and_unload()

        merged.save_pretrained(merged_path, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        tokenizer.save_pretrained(merged_path)


    validator = SingleGPUValidator(config, start_datetime)

    try:
        validator.validate()

    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        validator.cleanup()

        end_time = time.time()
        end_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        duration = end_time - start_time
        print(f"Validation ended: {end_datetime}")
        print(f"Total time: {duration/60:.2f} minutes")


if __name__ == "__main__":
    main()
