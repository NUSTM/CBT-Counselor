#!/usr/bin/env bash
set -Eeuo pipefail


partial_pretrain="/path/to/base_model"
project_name="CBT-Counselor-SFT"


epochs=3
lr="1e-4"


train_batch_size=32
micro_batch_size_per_gpu=16
val_batch_size_per_gpu=2


lora_rank=32
lora_alpha=64


PYTHON=${PYTHON:-python}
ENTRY=${ENTRY:-/path/to/verl/verl/trainer/fsdp_sft_trainer.py}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-12355}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

echo "Running pre-checks..."

if [[ ! -f "$ENTRY" ]]; then
  echo "ENTRY not found: $ENTRY"
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found, please install PyTorch distributed components"
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_ARR[@]}
if [[ "$GPU_COUNT" -ne "$NPROC_PER_NODE" ]]; then
  echo "CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' has $GPU_COUNT GPUs but NPROC_PER_NODE=$NPROC_PER_NODE"
  exit 1
fi

echo "Using devices: $CUDA_VISIBLE_DEVICES ($GPU_COUNT GPU(s))"
echo "Training params: epochs=$epochs, lr=$lr"

echo "Starting training..."

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_PORT="$MASTER_PORT" \
torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" --master-port="$MASTER_PORT" \
    "$ENTRY" \
    model.partial_pretrain="$partial_pretrain" \
    model.lora_rank=$lora_rank \
    model.lora_alpha=$lora_alpha \
    trainer.project_name="$project_name" \
    trainer.total_epochs=$epochs \
    optim.lr=$lr \
    data.train_batch_size=$train_batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.val_batch_size_per_gpu=$val_batch_size_per_gpu \
    "$@"

exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    echo "Training failed with exit code: $exit_code"
    exit $exit_code
fi

echo "Training complete"
