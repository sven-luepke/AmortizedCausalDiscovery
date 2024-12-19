#!/bin/bash
#SBATCH --o=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-hgx-a100-80x4  # Partition name
#SBATCH --ntasks=4                      # Number of tasks
#SBATCH --qos=mcml                      # Quality of service
#SBATCH --gres=gpu:1                    # Number of GPUs (if needed)
#SBATCH --container-image="nvcr.io/nvidia/pytorch:24.11-py3"
#SBATCH -D ./

pip install pylint==3.2.7
pip install seaborn==0.13.2
pwd
python -m data.generate_dataset
python -m train --suffix _springs5  # observed
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 0  # ACD with latent
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 1  # None
