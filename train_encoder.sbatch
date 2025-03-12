#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=train_encoder
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=train_encoder.out
#SBATCH --gres=gpu:1

module purge
cd /scratch/$USER
singularity exec --nv \
--overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "source /ext3/env.sh; conda activate fashion_env; cd /scratch/$USER/fashion-aesthetics-ml; python3 main.py --mode train_encoder --batch_size 32 --epochs 50 --lr 1e-4 --embedding_dim 768 --use_gpu"
