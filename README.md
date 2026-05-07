# Environment Setup

```bash
conda create -n verl python=3.10

conda activate verl

pip install --no-cache-dir "vllm==0.10.2" "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" tensordict torchdata

pip install flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install flashinfer_jit_cache-0.4.0.dev20251008+cu128-cp39-abi3-manylinux_2_28_x86_64.whl

pip install bert_score rouge swanlab nltk scikit-learn
```

---

# SFT Training & Validation

### SFT Training

```bash
cd verl

PYTHONPATH=/path/to/verl ./run_sft.sh >> SFT-training.log 2>&1 &

# Multi-GPU (e.g., 2 GPUs)
PYTHONPATH=/path/to/verl CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ./run_sft.sh >> SFT-training.log 2>&1 &
```

---

### SFT Model Evaluation

```bash
cd verl

PYTHONPATH=/path/to/verl nohup python verl/trainer/sft_validation_load_merged_model.py >> SFT-validation.log 2>&1 &
```

---

### GRPO Training

```bash
cd /path/to/verl
nohup ./run_ppo.sh >> GRPO-training.log 2>&1 &
echo "Training started, PID: $!"

# Monitor progress
# tail -f GRPO-training.log

# Stop training
# kill <PID>
```

---

### GRPO Model Evaluation

```bash
cd verl
PYTHONPATH=/path/to/verl nohup python verl/trainer/ppo_validation_load_pt_weight.py >> GRPO-validation.log 2>&1 &
```
