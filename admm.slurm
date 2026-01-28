#!/bin/bash
#SBATCH --job-name=admm_logreg
#SBATCH --partition=short-28core
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=00:20:00
#SBATCH --output=proj3_admm_%j.out
#SBATCH --error=proj3_admm_%j.err

module purge
module load gcc openmpi python

source ~/venvs/mpi/bin/activate

srun python admm_logreg_mpi.py \
  --data-dir /gpfs/projects/AMS598/projects2025_data/project3_data/ \
  --rho 1.0 \
  --max-iters 200
