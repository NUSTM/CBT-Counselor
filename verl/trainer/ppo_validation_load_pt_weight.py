#!/usr/bin/env python3


import glob
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel

from CBT_Counselor.evaluation_metric import MetricsCalculator
from verl.utils import hf_tokenizer
from verl.utils.dataset.ppo_counselor_dataset import PPOCounselorDataset, collate_fn
from verl.utils.fs import copy_to_local


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SimpleLoRAValidator:

    def __init__(self, config, validation_time):
        self.config = config
        self.validation_time = validation_time
        self.device = config.rollout.device

        if self.config.cbt_guided:
            self.model_name = self.config.model.partial_pretrain.split("/")[-1] + "_cbt_guided"
        else:
            self.model_name = self.config.model.partial_pretrain.split("/")[-1] + "_non_cbt_guided"

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

        print(f"Checkpoint: {self.config.model.checkpoint_path}")
        print(f"Validation samples: {len(self.val_dataset)}")

    def _load_tokenizer(self):
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        print(f"Tokenizer loaded: {local_model_path}")

    def _load_datasets(self):
        val_files = self.config.data.val_files
        print(f"Loading validation dataset: {val_files}")

        self.val_dataset = PPOCounselorDataset(
            parquet_files=val_files,
            tokenizer=self.tokenizer,
            config=self.config
        )

        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        print(f"Validation dataset loaded: {len(self.val_dataset)} samples")
        print(f"Validation batches: {len(self.val_dataloader)}")

    def _load_checkpoint_state_dict(self) -> dict:
        """Auto-detect checkpoint format and load state_dict.
        Supports single-GPU (world_size=1) and multi-GPU (world_size>=2) FSDP sharded checkpoints.
        """
        ckpt_dir = self.config.model.checkpoint_path
        pattern = os.path.join(ckpt_dir, "model_world_size_*_rank_*.pt")
        pt_files = glob.glob(pattern)

        if not pt_files:
            raise FileNotFoundError(f"No .pt checkpoint files found: {pattern}")

        def _sort_key(f):
            m = re.search(r"model_world_size_(\d+)_rank_(\d+)\.pt", os.path.basename(f))
            return (int(m.group(1)), int(m.group(2))) if m else (999, 999)
        pt_files = sorted(pt_files, key=_sort_key)
        match = re.search(r"model_world_size_(\d+)_rank_(\d+)\.pt", os.path.basename(pt_files[0]))
        if not match:
            raise ValueError(f"Cannot parse checkpoint filename: {pt_files[0]}")

        world_size = int(match.group(1))

        expected_files = sorted([
            os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{r}.pt")
            for r in range(world_size)
        ])
        for f in expected_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing checkpoint shard: {f}")

        if world_size == 1:
            print(f"Loading single-GPU .pt weights: {expected_files[0]}")
            return torch.load(expected_files[0], map_location="cpu", weights_only=False)

        print(f"Detected {world_size}-GPU checkpoint, merging shards...")
        return self._merge_fsdp_shards(expected_files)

    def _merge_fsdp_shards(self, pt_paths: list) -> dict:
        """Merge multi-GPU FSDP sharded checkpoints into a full state_dict.
        Supports both DTensor and plain tensor formats.
        """
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            try:
                from torch.distributed.tensor import DTensor
            except ImportError:
                DTensor = None

        shard_state_dicts = []
        for p in pt_paths:
            shard_state_dicts.append(torch.load(p, map_location="cpu", weights_only=False))

        all_keys = set(shard_state_dicts[0].keys())
        merged = {}

        for key in sorted(all_keys):
            tensors = [sd[key] for sd in shard_state_dicts]

            if DTensor is not None and isinstance(tensors[0], DTensor):
                placements = tuple(tensors[0].placements)
                if placements[0].is_replicate():
                    merged[key] = tensors[0]._local_tensor
                elif placements[0].is_shard():
                    dim = placements[0].dim
                    local_tensors = [t._local_tensor for t in tensors]
                    merged[key] = torch.cat(local_tensors, dim=dim).contiguous()
                else:
                    merged[key] = tensors[0]._local_tensor
            else:
                shapes = [t.shape for t in tensors]
                if len(set(str(s) for s in shapes)) == 1:
                    merged[key] = tensors[0]
                else:
                    merged[key] = torch.cat(tensors, dim=0).contiguous()

        del shard_state_dicts
        print(f"Merge complete, {len(merged)} parameters total")
        return merged

    def _load_model(self):
        print("Loading model...")

        base_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "bf16")
        if torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"Loading base model: {base_model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            trust_remote_code=self.config.model.trust_remote_code,
        )
        print("Base model loaded")

        state_dict = self._load_checkpoint_state_dict()

        _prefix = "_fsdp_wrapped_module."
        state_dict = {k[len(_prefix):] if k.startswith(_prefix) else k: v for k, v in state_dict.items()}

        embed_key = "base_model.model.model.embed_tokens.weight"
        if embed_key in state_dict:
            ckpt_vocab_size = state_dict[embed_key].shape[0]
            model_vocab_size = base_model.config.vocab_size
            print(f"Checkpoint vocab size: {ckpt_vocab_size}")
            print(f"Base model vocab size: {model_vocab_size}")

            if ckpt_vocab_size != model_vocab_size:
                print(f"Resizing model embedding: {model_vocab_size} -> {ckpt_vocab_size}")
                base_model.resize_token_embeddings(ckpt_vocab_size)

        base_state_dict = {}
        lora_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("base_model.model."):
                new_key = key[len("base_model.model."):]

                if "lora_A" in new_key or "lora_B" in new_key:
                    lora_state_dict[new_key] = value
                elif ".base_layer." in new_key:
                    new_key = new_key.replace(".base_layer.", ".")
                    base_state_dict[new_key] = value
                else:
                    base_state_dict[new_key] = value
            else:
                base_state_dict[key] = value

        print(f"Separated base weights: {len(base_state_dict)}")
        print(f"Separated LoRA weights: {len(lora_state_dict)}")

        missing, unexpected = base_model.load_state_dict(base_state_dict, strict=False)
        print(f"Base weights loaded - missing: {len(missing)}, unexpected: {len(unexpected)}")

        if missing:
            print(f"Missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"Unexpected keys (first 5): {unexpected[:5]}")

        print("Merging LoRA weights...")
        lora_alpha = self.config.model.lora_alpha
        lora_r = self.config.model.lora_rank
        scaling = lora_alpha / lora_r

        merged_count = 0
        base_model_state = base_model.state_dict()

        for lora_key, lora_value in lora_state_dict.items():
            if "lora_B" in lora_key:
                lora_a_key = lora_key.replace("lora_B", "lora_A")
                if lora_a_key in lora_state_dict:
                    lora_a = lora_state_dict[lora_a_key]
                    lora_b = lora_value

                    delta = (lora_b @ lora_a) * scaling

                    base_key = lora_key.replace(".lora_B.default.weight", ".weight")
                    base_key = base_key.replace(".lora_B.weight", ".weight")

                    if base_key in base_model_state:
                        base_model_state[base_key] = base_model_state[base_key] + delta
                        merged_count += 1

        print(f"Merged {merged_count} LoRA layers")
        base_model.load_state_dict(base_model_state)

        self.model = base_model

        print(f"Moving model to {self.device}...")
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded, device: {next(self.model.parameters()).device}")



    def validate(self):
        print("Starting single-GPU inference validation")

        results_file = self._setup_results_directory()

        self._run_validation_loop(results_file)

        if results_file:
            self._finalize_validation_results(results_file)

    def _setup_results_directory(self) -> Path:
        save_dir = Path(self.config.rollout.ppo_validation_files_path)

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

    def _run_validation_loop(self, results_file: Path):
        with torch.no_grad():
            for batch_idx, val_data in enumerate(tqdm(self.val_dataloader, desc="Validation")):

                sample_indices = val_data["index"]

                batch_references = []
                batch_last_speaker_utterances = []

                for sample_idx in sample_indices:
                    ref = self.val_dataset._get_gt_response(sample_idx)
                    last_speaker_utterance = self.val_dataset._get_last_spkeaker_utterance(sample_idx)
                    batch_references.append(ref)
                    batch_last_speaker_utterances.append(last_speaker_utterance)

                batch_predictions = self._process_batch(val_data)

                with open(results_file, 'a', encoding='utf-8') as f:
                    for pred, ref, last_speaker_utterance in zip(batch_predictions, batch_references, batch_last_speaker_utterances):
                        sample_data = {
                            'prediction': pred,
                            'reference': ref,
                            'last_speaker_utterance': last_speaker_utterance
                        }
                        f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')

    def _process_batch(self, val_data: Dict) -> List[str]:
        input_ids = val_data["input_ids"]
        attention_mask = val_data["attention_mask"]

        if not input_ids.is_cuda and self.device.startswith("cuda"):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        if "repetition_penalty" in self.config.rollout:
            # for CBT-Qwen3
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
            # for CBT-Llama3.2
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

        generated_texts = self._extract_generated_tokens(outputs, input_ids)

        return generated_texts

    def _extract_generated_tokens(self, full_outputs: torch.Tensor, original_input_ids: torch.Tensor) -> List[str]:
        batch_size = full_outputs.shape[0]
        original_seq_len = original_input_ids.shape[1]

        generated_texts = []

        for i in range(batch_size):
            if full_outputs.shape[1] > original_seq_len:
                generated_tokens = full_outputs[i, original_seq_len:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""

            generated_texts.append(generated_text.strip())

        return generated_texts


    def _finalize_validation_results(self, results_file: Path):
        print("Computing final metrics...")
        final_metrics = self._compute_metrics_from_file(results_file)

        results_file = Path(results_file)

        eval_set_name = self.config.data.val_files.split("/")[-1].split("_")[0]
        experi_setting = self.config.model.checkpoint_path.split("/")[-2]
        metrics_file = results_file.parent / f"{self.model_name}_{experi_setting}_{eval_set_name}_{self.validation_time}_metrics.json"

        if metrics_file.exists():
            metrics_file.unlink()

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
            print(f"Final metrics saved to: {metrics_file}")

    def _compute_metrics_from_file(self, results_file: Path) -> Dict:
        all_references = []
        all_predictions = []
        all_last_speaker_utterances = []

        print(f"Reading results from file: {results_file}")

        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample_data = json.loads(line)
                    all_references.append(sample_data["reference"])
                    all_predictions.append(sample_data["prediction"])
                    all_last_speaker_utterances.append(sample_data["last_speaker_utterance"])

        metrics = self.metrics_calculator._compute_evaluation_metrics(
            all_references, all_predictions, all_last_speaker_utterances, self.config.cbt_guided
        )

        return metrics



    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def main():
    import argparse

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(f"Validation started: {start_datetime}")

    parser = argparse.ArgumentParser(description="LoRA model validation")

    parser.add_argument(
        "--config_path",
        type=str,
        default="/path/to/CBT_Counselor/counselor_configs/ppo_val_config.yaml"
    )


    args = parser.parse_args()


    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    config = OmegaConf.load(args.config_path)

    validator = SimpleLoRAValidator(config, start_datetime)

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
