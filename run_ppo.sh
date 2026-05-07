#!/bin/bash
set -x

# actual_train_batch_size = train_batch_size × actor_rollout_ref.rollout.n
# mini_batch_size ÷ n_gpus_per_node must be an integer
# ppo_micro_batch_size_per_gpu × n_gpus_per_node ≤ ppo_mini_batch_size

train_path="/path/to/rl_training_data.parquet"
val_path="/path/to/rl_validation_data.parquet"

load_sft_checkpoint_merged="/path/to/checkpoints/sft/merged_model"

reward_function_path="/path/to/verl/CBT_Counselor/PPO_reward_function.py"

checkpoint_dir="/path/to/checkpoints/grpo"

# Resume from checkpoint:
# ++trainer.resume=true \
# ++trainer.resume_mode=resume_path \
# +trainer.resume_from_path="/path/to/checkpoints/grpo/global_step_30" \
# ++trainer.del_local_ckpt_after_load=false \

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_path" \
    data.val_files="$val_path" \
    data.train_batch_size=64 \
    actor_rollout_ref.model.path=${load_sft_checkpoint_merged} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=6 \
    reward_model.custom_reward_function.reward_kwargs.group_size=6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.actor.checkpoint.save_interval=20 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.log_val_generations=20 \
    trainer.default_local_dir="$checkpoint_dir" "$@"
