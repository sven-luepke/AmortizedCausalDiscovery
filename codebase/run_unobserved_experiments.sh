#!/bin/bash
#SBATCH --output=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-hgx-a100-80x4  # Partition name
#SBATCH --ntasks=4                      # Number of tasks
#SBATCH --qos=mcml                      # Quality of service
#SBATCH --gres=gpu:1                    # Number of GPUs (if needed)
#SBATCH --container-image="nvcr.io/nvidia/pytorch:24.11-py3"
#SBATCH -D ./

pip install pylint==3.2.7 seaborn==0.13.2
cd ~/workspace/AmortizedCausalDiscovery/codebase
pwd
python -m test
python -m data.generate_dataset
python -m train --suffix _springs5 # observed
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 0  # ACD with latent
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 1  # None
python -m train --suffix _springs5 --unobserved 1 --model_unobserved 2  # Mean

python -m data.generate_dataset --uninfluenced_particle --influencer_particle
python -m train --suffix _springs5_uninfluenced_influencer  # observed
python -m train --suffix _springs5_uninfluenced_influencer --unobserved 1 --model_unobserved 0  # ACD with latent
python -m train --suffix _springs5_uninfluenced_influencer --unobserved 1 --model_unobserved 1  # None
python -m train --suffix _springs5_uninfluenced_influencer --unobserved 1 --model_unobserved 2  # Mean