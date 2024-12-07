#!/bin/bash -l
######################### Batch Headers #########################
#SBATCH --partition chengji-lab-gpu  # NOTE: use reserved partition `chengji-lab-gpu` to use reserved A100 or H100 GPUs
#SBATCH --account chengji-lab  # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres gpu:H100:4      # request A100 GPU resource(s)
#SBATCH --ntasks-per-node=4    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=0                # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t 7-00:00:00          # time limit for the job (up to 7 days: `7-00:00:00`)
#SBATCH -J esmfold_prior_training # job name
#SBATCH --output=R-%x.%j.out   # output log file
#SBATCH --error=R-%x.%j.err    # error log file

random_seconds=$(( (RANDOM % 100) + 1 ))
echo "Sleeping for $random_seconds seconds before starting run"
sleep "$random_seconds"

module purge
module load cuda/11.8.0_gcc_9.5.0

# determine location of the project directory
use_private_project_dir=false # NOTE: customize as needed
if [ "$use_private_project_dir" = true ]; then
    project_dir="/home/acmwhb/data/Repositories/Lab_Repositories/FlowDock"
else
    project_dir="/cluster/pixstor/chengji-lab/acmwhb/Repositories/Lab_Repositories/FlowDock"
fi

# shellcheck source=/dev/null
source /cluster/pixstor/chengji-lab/acmwhb/miniforge3/etc/profile.d/conda.sh
conda activate "$project_dir"/FlowDock/

echo "Calling flowdock/train.py!"
cd "$project_dir" || exit
srun python3 flowdock/train.py \
    experiment='flowdock_fm' \
    environment=slurm \
    logger=wandb \
    logger.wandb.entity='bml-lab' \
    logger.wandb.group='FlowDock-FM' \
    +logger.wandb.name='2024-12-06_18:00:00-ESMFold-Prior-Training' \
    +logger.wandb.id='z0u52tvj' \
    model.cfg.prior_type=esmfold \
    model.cfg.task.freeze_score_head=false \
    model.cfg.task.freeze_affinity=false \
    strategy=ddp \
    trainer=ddp \
    trainer.devices=4 \
    trainer.num_nodes=1
echo "Finished calling flowdock/train.py!"

# NOTE: the following commands must be used to resume training from a checkpoint
# ckpt_path="$(realpath 'logs/train/runs/2024-05-17_13-45-06/checkpoints/last.ckpt')" \
# paths.output_dir="$(realpath 'logs/train/runs/2024-05-17_13-45-06')" \

# NOTE: the following commands may be used to speed up training
# model.compile=false \
# +trainer.precision=bf16-mixed
