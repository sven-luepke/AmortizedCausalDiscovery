#!/bin/bash
#SBATCH --output=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-dgx-a100-40x8  # Partition name
#SBATCH --ntasks=4                      # Number of tasks
#SBATCH --qos=mcml                      # Quality of service
#SBATCH --gres=gpu:1                    # Number of GPUs (if needed)
#SBATCH --container-image="nvcr.io/nvidia/pytorch:24.11-py3"
#SBATCH -D ./
#SBATCH -t 2-00:00:00

pip install pylint==3.2.7 seaborn==0.13.2
cd ~/workspace/AmortizedCausalDiscovery/codebase
pwd

python -u -m train --suffix _springs5 --unobserved 1 --model_unobserved 0 --dont_shuffle_unobserved --epochs 8  # ACD with latent
python -u -m train --suffix _springs5 --unobserved 1 --model_unobserved 3 --dont_shuffle_unobserved --epochs 8 # New model