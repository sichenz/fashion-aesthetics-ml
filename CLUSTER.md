## NYU Greene Cluster

### Setting Up Singularity Container (ONLY DO THIS ONCE)

```bash
# List available overlay file systems
ls /scratch/work/public/overlay-fs-ext3

# Navigate to user scratch directory
cd /scratch/$USER

# Copy and decompress the overlay file
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
gunzip overlay-50G-10M.ext3.gz

# List available Singularity images
ls /scratch/work/public/singularity/

# Execute Singularity container
singularity exec --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
```

### Installing Conda Inside the Container

```bash
# Download and install Miniforge
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
rm Miniforge3-Linux-x86_64.sh
```

### Setting Up Environment Variables

```bash
touch /ext3/env.sh
nano /ext3/env.sh
```

Add the following lines to `env.sh`:

```bash
#!/bin/bash

unset -f which

source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
```

Then, source the environment:

```bash
source /ext3/env.sh
```

### Configuring Conda

```bash
conda config --remove channels defaults  # Ignore CondaKeyError warning
conda update -n base conda -y
conda clean --all --yes
conda install pip -y
conda install ipykernel -y
```

---

## Creating and Activating a New Conda Environment

```bash
cd /scratch/$USER

# Use previous Singularity session
singularity exec --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

# Load environment variables
source /ext3/env.sh

# Create a new environment
conda create -n <environment_name> python=<python_version>
conda activate <environment_name>
```

---

## Installing Required Packages

```bash
cd /scratch/$USER

singularity exec --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

source /ext3/env.sh
conda activate <environment_name>
cd /home/$USER/<project_directory>
pip install -r requirements.txt
exit
```

---

## Submitting Jobs with SLURM

Create a job submission script (`<job_name>.sbatch`):

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=<job_name>
#SBATCH --mail-user=<your_email>
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=<job_name>.out
#SBATCH --gres=gpu:1

module purge
cd /scratch/$USER
singularity exec --nv \
    --overlay /scratch/$USER/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate <environment_name>; cd /home/$USER/<project_directory>; python3 main.py"
```

Submit the job:

```bash
sbatch <job_name>.sbatch
```

Check job status:

```bash
squeue -u $USER
```

