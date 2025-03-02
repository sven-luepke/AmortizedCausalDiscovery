#!/bin/bash
#SBATCH --output=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-hgx-a100-80x4  # Partition name
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

python -m data.generate_dataset --dynamic
python -m train --suffix _springs5_dynamic --epochs=500 --encoder=transformer --lr=1e-4
python -m train --suffix _springs5_dynamic --epochs=500 --encoder=mlp --lr=1e-4

#python -u -m train --suffix _springs5 --epochs=128 --encoder=transformer --encoder_steps=0 --lr=1e-4
#python -u -m train --suffix _springs5 --epochs=128 --encoder=transformer --encoder_steps=8 --lr=1e-4

# TODO: transformer encoder from the amortized inference paper
#python -u -m train --suffix _springs5 --epochs=100 --encoder=transformer