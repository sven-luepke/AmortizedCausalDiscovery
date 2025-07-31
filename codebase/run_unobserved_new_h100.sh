#!/bin/bash
#SBATCH --output=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-hgx-h100-92x4  # Partition name
#SBATCH --ntasks=2                      # Number of tasks
#SBATCH --qos=mcml                      # Quality of service
#SBATCH --gres=gpu:1                    # Number of GPUs (if needed)
#SBATCH --container-image="nvcr.io/nvidia/pytorch:24.11-py3"
#SBATCH -D ./
#SBATCH -t 2-00:00:00

pip install pylint==3.2.7 seaborn==0.13.2
pip list
cd ~/workspace/AmortizedCausalDiscovery/codebase
pwd

python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 3 --predict_initial_point --exclude_loss_unobserved
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 3 --predict_initial_point
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 3 --exclude_loss_unobserved
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 3 
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 0
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 1
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200 --unobserved 1 --model_unobserved 2
python -u -m train --suffix _springs5 --dont_shuffle_unobserved --epochs=200