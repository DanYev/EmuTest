#!/bin/bash
#SBATCH --time=0-02:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -G a100:1
#SBATCH -o slurm_jobs/output.%A.%a.out
#SBATCH -e slurm_jobs/error.%A.%a.err

PYSCRIPT="$@"   

python $PYSCRIPT
