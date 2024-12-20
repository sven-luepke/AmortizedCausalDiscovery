#!/bin/bash
#SBATCH --output=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-hgx-a100-80x4  # Partition name
#SBATCH --ntasks=4                      # Number of tasks
#SBATCH --qos=mcml                      # Quality of service
#SBATCH --gres=gpu:1                    # Number of GPUs (if needed)
#SBATCH --container-image="nvcr.io/nvidia/pytorch:24.11-py3"
#SBATCH -D ./
#SBATCH -t 2-00:00:00

pip install pylint==3.2.7 seaborn==0.13.2
cd ~/workspace/AmortizedCausalDiscovery/codebase
pwd

# Table 4
python -u -m data.generate_dataset --confounder
python -u -m train --suffix _springs5_conf # observed
python -u -m train --suffix _springs5_conf --unobserved 1 --model_unobserved 0 --dont_shuffle_unobserved  # ACD with latent
python -u -m train --suffix _springs5_conf --unobserved 1 --model_unobserved 1 --dont_shuffle_unobserved  # None
python -u -m train --suffix _springs5_conf --unobserved 1 --model_unobserved 2 --dont_shuffle_unobserved  # Mean

# Table 5
python -u -m data.generate_dataset --uninfluenced_particle --influencer_particle
python -u -m train --suffix _springs5_uninfluenced_influencer  # observed
python -u -m train --suffix _springs5_uninfluenced_influencer --unobserved 1 --model_unobserved 0 --dont_shuffle_unobserved  # ACD with latent
python -u -m train --suffix _springs5_uninfluenced_influencer --unobserved 1 --model_unobserved 1 --dont_shuffle_unobserved  # None
python -u -m train --suffix _springs5_uninfluenced_influencer --unobserved 1 --model_unobserved 2 --dont_shuffle_unobserved  # Mean