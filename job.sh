#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --mem=15G                        # Ask for 2 GB of RAM
#SBATCH --gres=gpu:2              # Number of GPUs (per node)
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00                   # The job will run for 3 hours
#SBATCH --output=./OUT/tabular-%j.out
#SBATCH --mail-user=anand.kamat@mail.mcgill.ca
#SBATCH --mail-type=END

# 1. Create your environement locally
module load python/3.6
module load cuda cudnn 
source ~/PPOC_gpu/bin/activate


python ./baselines/ppo1/run_mujoco.py --saves --opt=4 --minibatch=200 --dc=0.1 --tradeoff=0.01 --prew_control=1e3 --caption='' --diayn --seed=11
