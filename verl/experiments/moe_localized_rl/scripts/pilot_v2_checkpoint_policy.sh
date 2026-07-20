# Shared pilot v2 checkpoint / resume policy.
# Source from run_pilot_*.sh — do not execute directly.

# Save every 10 steps; keep only the latest checkpoint on disk.
export SAVE_FREQ=${SAVE_FREQ:-10}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export RESUME_MODE=${RESUME_MODE:-auto}

# Disable mid-training validation (TransferQueue crash on MATH val set).
export TEST_FREQ=${TEST_FREQ:--1}
# veRL reads trainer.val_before_train (not test_before_train).
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# Model weights only (~14GB/ckpt). Trainer still saves data.pt + transfer_queue
# alongside each global_step_* folder for resume.
export CKPT_SAVE_CONTENTS=${CKPT_SAVE_CONTENTS:-"['model']"}

# Checkpoint-related Hydra overrides (append to TRAINER array in run scripts).
pilot_v2_checkpoint_trainer_args() {
    echo \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
        trainer.resume_mode=${RESUME_MODE}
}
