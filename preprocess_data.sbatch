#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=preprocess_data
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=preprocess_data.out
#SBATCH --gres=gpu:1

module purge
cd /scratch/$USER
singularity exec --nv \
--overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "source /ext3/env.sh; conda activate fashion_env; cd /scratch/$USER/fashion-aesthetics-ml; python3 main.py --preprocess_data"
