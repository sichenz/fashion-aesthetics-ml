#!/bin/bash
# sbatch_train_generator.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=train_generator
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=train_generator.out
#SBATCH --gres=gpu:1

module purge
cd /scratch/$USER
singularity exec --nv \
--overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "source /ext3/env.sh; conda activate fashion_env; cd /scratch/$USER/fashion-aesthetics-ml; python3 main.py --mode train_generator --use_gpu"
