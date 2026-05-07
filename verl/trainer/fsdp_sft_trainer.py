# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer with Multi-task Learning
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.distributed
import torch.distributed as dist
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

import verl.utils.hdfs_io as hdfs_io
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path, get_checkpoint_tracker_filename)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.sft_counselor_dataset import SFTDataset
from verl.utils.device import (get_device_id, get_device_name,
                               is_cuda_available, is_npu_available)
from verl.utils.distributed import (destroy_global_process_group,
                                    initialize_global_process_group)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (get_fsdp_wrap_policy,
                                   get_init_weight_context_manager, init_fn)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import (get_cosine_schedule_with_warmup,
                                         get_wsd_schedule_with_warmup)
from verl.utils.tracking import Tracking
from verl.workers.sharding_manager.fsdp_ulysses import \
    FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None



class FSDPSFTTrainer:

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer


        if self.config.cbt_guided:
            self.sft_model_name = self.config.model.partial_pretrain.split("/")[-1] + "_cbt_guided"
        else:
            self.sft_model_name = self.config.model.partial_pretrain.split("/")[-1] + "_no_cbt_guided"

        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")
        
        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)

        self._build_dataloader(train_dataset, val_dataset)

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device


    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.global_batch_size = self.config.data.train_batch_size

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        device_name = get_device_name()

        # Get num_workers from config, default to 8
        num_workers = getattr(config.data, 'num_workers', 8)

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        # Validation set is not shuffled to preserve dialogue ground-truth order for metric computation
        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.val_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_model_optimizer(self):

        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Step 1: Load config
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config

        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            if self.device_mesh.get_rank() == 0:
                print(f"Loading {self.sft_model_name} model...")

            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            
            if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
                if self.device_mesh.get_rank() == 0:
                    print(f"Resizing embeddings: from {self.model.get_input_embeddings().num_embeddings} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=128)
                
                if self.device_mesh.get_rank() == 0:
                    print(f"Embeddings resized successfully, new size: {self.model.get_input_embeddings().num_embeddings}")

            if self.device_mesh.get_rank() == 0:
                print(f"{self.sft_model_name} model loaded successfully")


            # Step 2: Apply LoRA to the entire model
            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()

                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }

                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
            
            self.model = self.model.to(torch_dtype)
        
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        log_gpu_memory_usage("After model allocation", logger=logger)

        # Set mixed precision policy
        mixed_precision = MixedPrecision(
            param_dtype=torch_dtype, 
            reduce_dtype=torch.float32, 
            buffer_dtype=torch.float32
        )
        
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,  
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        # Set CPU offload
        cpu_offload = None
        if self.config.model.fsdp_config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        # Create FSDP model
        self.fsdp_model = FSDP(
            self.model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # Compute training steps and learning rate scheduler
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )
        
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=self.total_steps
            )

    def _compute_loss_and_backward(self, batch, do_backward=True):
        
        """Compute loss"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        
        # Context manager for sequence parallel
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            labels = input_ids[:, 1:].contiguous()
            
            output = self.fsdp_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                use_cache=False
            )
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.contiguous()
            shift_logits = shift_logits.view(-1, logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            lm_loss = loss_fct(shift_logits, shift_labels)
            lm_loss = lm_loss * loss_mask.to(lm_loss.device)
            loss_info = {"lm_loss": lm_loss}


        # Compute final loss and metrics
        valid_token_this_rank = torch.sum(loss_mask)
        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)
            dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
        else:
            dp_size = 1

        # Normalize language modeling loss
        lm_loss_normalized = torch.sum(loss_info["lm_loss"]) / (valid_token_this_rank + 1e-8) * dp_size

        total_loss = lm_loss_normalized
        loss_info = {"total_loss": total_loss.item()}

        if do_backward:
            total_loss.backward()
        
        return total_loss, loss_info


    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
        self.optimizer.zero_grad()
        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        step_loss_info = {}
        
        for micro_batch in micro_batches:
            loss, loss_info = self._compute_loss_and_backward(batch=micro_batch)
            step_loss += loss.item() / n_micro_batches
            
            # Accumulate loss info
            for key, value in loss_info.items():
                if key not in step_loss_info:
                    step_loss_info[key] = 0
                step_loss_info[key] += value / n_micro_batches

        # Gradient clipping
        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)
        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]
        log_gpu_memory_usage("After lr scheduler step", logger=logger)

        # Prepare metrics for logging
        metrics = {
            "train/loss": step_loss,
            "train/lr(1e-3)": lr * 1e3
        }
        
        # Reduce metrics across dp ranks
        for key, value in metrics.items():
            if key != "train/lr(1e-3)":  # Don't reduce learning rate
                value_tensor = torch.tensor(value).to(self.device_name)
                torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.AVG)
                metrics[key] = value_tensor.detach().item()

        return metrics


    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, loss_info = self._compute_loss_and_backward(batch, do_backward=False)
            
            # Reduce losses across dp ranks
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
            
            return loss


    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        if SFTDataset.get_thinking_flag():
            folder = "thinking_mode"
        else:
            folder = "no_thinking_mode"
        
        if self.config.data.subset is not None:
            subset = self.config.data.subset
        else:
            subset = ""
            
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, 
            folder,
            subset,
            f"{self.sft_model_name}_Epoch_{self.config.trainer.total_epochs}_Lr_{self.config.optim.lr}", 
            f"global_step_{step}"
        )


        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, 
            global_step=step, 
            max_ckpt_to_keep=max_ckpt_to_keep
        )

        # LoRA saving logic
        if self.config.model.get("lora_rank", 0) > 0:
            lora_save_path = os.path.join(local_global_step_folder, "lora_adapter") 
            
            if self.device_mesh.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
            
            # Get PEFT model
            if hasattr(self.fsdp_model, '_fsdp_wrapped_module'):
                peft_model = self.fsdp_model._fsdp_wrapped_module
            else:
                peft_model = self.fsdp_model
                
            if hasattr(peft_model, "peft_config"):
                try:
                    # Extract LoRA parameters
                    from verl.utils.fsdp_utils import layered_summon_lora_params
                    from safetensors.torch import save_file
                    from dataclasses import asdict
                    import json
                    
                    lora_params = layered_summon_lora_params(self.fsdp_model)
                    
                    if self.device_mesh.get_rank() == 0:
                        # Save LoRA weights
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        
                        # Save config
                        peft_config = asdict(peft_model.peft_config.get("default", {}))
                        peft_config["task_type"] = peft_config["task_type"].value
                        peft_config["peft_type"] = peft_config["peft_type"].value
                        peft_config["target_modules"] = list(peft_config["target_modules"])
                        
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w") as f:
                            json.dump(peft_config, f, indent=4)
                        
                        print(f"LoRA adapter saved to: {lora_save_path}")
                            
                except Exception as e:
                    print(f"Save LoRA Adapter Error: {e}")
            
            torch.distributed.barrier()

        # Save dataloader state (after LoRA saving)
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()
        
        return local_global_step_folder



    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager with the FSDP model
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )
        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")  # disable
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def _log_hyperparameters(self, tracking, use_swanlab):
        """Log training hyperparameters to the logging system"""
        try:
            # Build hyperparameter dict
            hyperparams = {
                # Training config
                "config/total_epochs": self.config.trainer.total_epochs,
                "config/total_training_steps": getattr(self, 'total_training_steps', None),
                "config/steps_per_epoch": getattr(self, 'steps_per_epoch', None),

                # Optimizer config
                "config/learning_rate": self.config.optim.lr,
                "config/warmup_steps_ratio": getattr(self.config.optim, 'warmup_steps_ratio', 0.0),
                "config/weight_decay": getattr(self.config.optim, 'weight_decay', 0.0),
                "config/clip_grad": getattr(self.config.optim, 'clip_grad', 1.0),
                
                # Data config
                "config/global_batch_size": self.global_batch_size,
                "config/micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
                "config/max_length": self.config.data.max_length,
                "config/balance_dp_token": getattr(self.config.data, 'balance_dp_token', False),
                
                # Model config
                "config/lora_rank": getattr(self.config.model, 'lora_rank', 0),
                "config/lora_alpha": getattr(self.config.model, 'lora_alpha', None),

                # Distributed config
                "config/world_size": self.device_mesh.size(),
                "config/dp_size": self.device_mesh.size(0) if not hasattr(self, 'ulysses_device_mesh') else self.ulysses_device_mesh.size(0),
            }

            # Add dataset info
            if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, '__len__'):
                hyperparams["config/train_dataset_size"] = len(self.train_dataset)
            if hasattr(self, 'val_dataset') and hasattr(self.val_dataset, '__len__'):
                hyperparams["config/val_dataset_size"] = len(self.val_dataset)
                
            # Filter out None values
            hyperparams = {k: v for k, v in hyperparams.items() if v is not None}
            
            # Log to tracking system
            if use_swanlab:
                try:
                    import swanlab

                    # SwanLab supports logging hyperparameters
                    swanlab.config.update(hyperparams)
                    print("Hyperparameters logged to SwanLab")
                except Exception as e:
                    print(f"SwanLab hyperparameter logging failed: {e}")
            
            # Also use tracking as backup
            if tracking:
                # Log all hyperparameters at step 0
                tracking.log(data=hyperparams, step=0)
                
            # Print hyperparameter summary
            print("Training hyperparameter configuration:")
            print(f"   total_epochs: {hyperparams.get('config/total_epochs')}")
            print(f"   learning_rate: {hyperparams.get('config/learning_rate')}")
            print(f"   global_batch_size: {hyperparams.get('config/global_batch_size')}")
            print(f"   max_sequence_length: {hyperparams.get('config/max_length')}")
            print(f"   world_size: {hyperparams.get('config/world_size')}")
            print(f"   LoRA config: rank={hyperparams.get('config/lora_rank')}")
            print(f"   fsdp_model_dtype: {self.config.model.fsdp_config.model_dtype}")
            print(f"   train_dataset_size: {hyperparams.get('config/train_dataset_size')}")
            print(f"   val_dataset_size: {hyperparams.get('config/val_dataset_size')}")
            
        except Exception as e:
            print(f"Hyperparameter logging failed: {e}")


    def fit(self):
        try:
            rank = self.device_mesh.get_rank()

            # Only rank0 uses Tracking
            tracking = None
            use_swanlab = False
            swanlab_run = None  # Track SwanLab run

            best_val_loss = float('inf')
            best_model_path = None

            if rank == 0:
                # SwanLab initialization
                if self.config.trainer.logger == "swanlab":
                    try:
                        import swanlab

                        # Check if there is already an active run
                        if not hasattr(swanlab, '_current_run') or swanlab._current_run is None:
                            experiment_name = self.config.trainer.experiment_name + f"_Epoch_{self.config.trainer.total_epochs}_LR_{self.config.optim.lr}"

                            swanlab_run = swanlab.init(
                                project=self.config.trainer.project_name,
                                experiment_name=experiment_name,
                                reinit=True  # Allow re-initialization
                            )
                        use_swanlab = True
                    except Exception as e:
                        log_with_rank(
                            f"SwanLab initialization failed: {e}, falling back to console logging",
                            logger=logger,
                            rank=0,
                            level=logging.WARNING,
                            log_only_rank_0=True
                        )
                        use_swanlab = False

                experiment_name = self.config.trainer.experiment_name + f"_Epoch_{self.config.trainer.total_epochs}_LR_{self.config.optim.lr}"

                # Create a Tracking instance that does not auto-close SwanLab
                tracking = Tracking(
                    project_name=self.config.trainer.project_name,
                    experiment_name=experiment_name,
                    default_backend=("console"),  # Always use console; manage SwanLab manually
                )

                # Manually log to SwanLab (if enabled)
                self._log_hyperparameters(tracking, use_swanlab)

            global_step = self.resume_global_step  # Start from resumed step
            last_valid_metric = None
            
            # Compute total training steps
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

            if self.config.trainer.total_training_steps is not None:
                total_training_steps = self.config.trainer.total_training_steps

            self.total_training_steps = total_training_steps
            
            # Get validation interval config; default to twice per epoch (steps_per_epoch // 2)
            eval_steps = getattr(self.config.trainer, 'eval_steps', None)
            if eval_steps is None:
                eval_steps = len(self.train_dataloader) // 2  # Default: validate twice per epoch
            
            log_with_rank(
                f"Total training steps: {self.total_training_steps}, Evaluation every {eval_steps} steps",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

            # Calculate which epoch we're starting from for sampler.set_epoch()
            start_epoch = global_step // self.steps_per_epoch

            # Start training
            for epoch in range(start_epoch, self.config.trainer.total_epochs):
                self.train_sampler.set_epoch(epoch=epoch)

                for step_in_epoch, data in enumerate(
                    tqdm(
                        self.train_dataloader,
                        initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                        total=self.steps_per_epoch,
                        desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                        disable=rank != 0,
                    )
                ):
                    global_step += 1
                    data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                    metric = self.training_step(data)
                    if rank == 0:
                        tracking.log(data=metric, step=global_step)
                        # Also log to SwanLab
                        if use_swanlab:
                            try:
                                import swanlab
                                swanlab.log(metric, step=global_step)
                            except Exception as e:
                                print(f"SwanLab logging failed: {e}")

                    is_last_step = global_step >= self.total_training_steps
                    
                    # Use configurable validation interval
                    is_valid_step = (global_step % eval_steps == 0) or is_last_step
                    
                    # Validation step
                    if is_valid_step:
                        
                        should_save = False  # Initialize only during validation
                        
                        # Run validation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        val_losses = []

                        self._current_val_step = 0
                        
                        is_main_process = (self.device_mesh.get_rank() == 0)

                        if is_main_process:
                            print(f"\nStarting validation (step {global_step}), total {len(self.val_dataloader)} batches...")

                        # Iterate over validation set
                        for val_data in self.val_dataloader:
                            val_data = TensorDict(val_data, batch_size=self.config.data.val_batch_size_per_gpu).to(self.device_name)
                        
                            val_loss = self.validation_step(val_data)
                            
                            val_losses.append(val_loss)
                            self._current_val_step += 1

                            if is_main_process and self._current_val_step % 10 == 0:
                                progress = self._current_val_step / len(self.val_dataloader) * 100
                                print(f"Validation progress: {self._current_val_step}/{len(self.val_dataloader)} ({progress:.1f}%)")

                            if self._current_val_step % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        if is_main_process:
                            # Process validation losses
                            val_loss = torch.mean(torch.stack(val_losses))
                            
                            current_val_loss = float(val_loss.detach().item())

                            val_metric = {"val/loss": float(val_loss.detach().item())}
                            
                            print(f"Validation complete (step {global_step}): avg_loss = {val_metric['val/loss']:.4f}")
        
                            # Log to tracking
                            tracking.log(data=val_metric, step=global_step)
                            # Also log to SwanLab
                            if use_swanlab:
                                try:
                                    import swanlab
                                    swanlab.log(val_metric, step=global_step)
                                except Exception as e:
                                    print(f"SwanLab validation logging failed: {e}")

                            # Check whether to save best model
                            if current_val_loss < best_val_loss:
                                print(f"New best model found! Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}")
                                
                                # Delete previous best model
                                if best_model_path is not None and os.path.exists(best_model_path):
                                    try:
                                        if os.path.isdir(best_model_path):
                                            import shutil
                                            shutil.rmtree(best_model_path)
                                            print(f"Deleted previous best model: {best_model_path}")
                                        else:
                                            os.remove(best_model_path)
                                            print(f"Deleted previous best model: {best_model_path}")
                                    except Exception as e:
                                        print(f"Error deleting previous model: {e}")
                                
                                # Update best validation loss
                                best_val_loss = current_val_loss
                                should_save = True

                            last_valid_metric = val_metric

                        should_save_tensor = torch.tensor(int(should_save), dtype=torch.long, device=self.device_name)
                        torch.distributed.broadcast(should_save_tensor, src=0)
                        should_save = bool(should_save_tensor.item())    
                        torch.distributed.barrier()
                    
                    # Save model
                    if is_last_step or (is_valid_step and should_save):
                        if rank == 0:
                            print(f"Saving checkpoint (step {global_step})")
                        
                        saved_path = self.save_checkpoint(step=global_step)
                        
                        if should_save and rank == 0:
                            best_model_path = saved_path
                            print(f"Best model saved: {saved_path}")

                    if is_last_step:
                        if rank == 0:
                            print(f"Final validation metrics: {last_valid_metric}")
                        break 
                
                # Break outer loop if all training steps are complete
                if global_step >= self.total_training_steps:
                    break
        
        except Exception as e:
            # Close SwanLab on exception
            if rank == 0 and use_swanlab:
                try:
                    import swanlab
                    if hasattr(swanlab, '_current_run') and swanlab._current_run is not None:
                        swanlab.finish()
                        print("SwanLab session closed on exception")
                except:
                    pass
            raise e
        
        finally:
            if rank == 0:
                try:
                    # Close SwanLab first
                    if use_swanlab and swanlab_run is not None:
                        import swanlab
                        if hasattr(swanlab, '_current_run') and swanlab._current_run is not None:
                            swanlab.finish()
                            print("SwanLab session closed normally")
                    
                    # Then clean up tracking (set to None to avoid auto-destructor issues)
                    if tracking is not None:
                        # Manually clean up tracking to prevent it from closing an already-closed SwanLab on destruction
                        if hasattr(tracking, 'logger') and 'swanlab' in tracking.logger:
                            tracking.logger['swanlab'] = None
                        del tracking
                        
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {cleanup_error}")
                    pass



def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    # Load base model from local path
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)  
    
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    # Prepare training and validation datasets
    if rank == 0:
        print(f"Loading SFT training set {config.data.train_files}...")
    train_dataset = SFTDataset(parquet_files=config.data.train_files, tokenizer=tokenizer, config=config, mode="train")
    
    if rank == 0:
        print(f"Loading SFT development set {config.data.val_files}...")
    val_dataset = SFTDataset(parquet_files=config.data.val_files, tokenizer=tokenizer, config=config, mode="train")

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    # Print message before destroying process group
    if rank == 0:
        print("SFT training complete. Please run the validation inference code to evaluate this checkpoint on automatic metrics.")
        print("*"*80)

    # Clean up resources
    if hasattr(trainer, 'model') and trainer.model is not None:
        # Special handling required for FSDP models
        if hasattr(trainer.model, '_fsdp_wrapped_module'):
            trainer.model = None
        del trainer.model
    
    if hasattr(trainer, 'model') and trainer.model is not None:
        del trainer.model

    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        del trainer.optimizer
        
    del trainer
    del train_dataset
    del val_dataset
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    destroy_global_process_group()




# Set environment variables
for k, v in {
    'RANK': '0',
    'LOCAL_RANK': '0',
    'WORLD_SIZE': '1',
    'MASTER_ADDR': 'localhost',
    'MASTER_PORT': '12356',
}.items():
    os.environ.setdefault(k, v)


@hydra.main(config_path="CBT_Counselor/counselor_configs",
            config_name="sft_train_config",
            version_base=None)
def main(config):
    run_sft(config)


if __name__ == "__main__":
    main()