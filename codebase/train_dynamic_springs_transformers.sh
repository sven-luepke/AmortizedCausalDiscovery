#!/bin/bash
#SBATCH --output=output_%j.log          # Standard output and error log (%j will be replaced by job ID)
#SBATCH --error=error_%j.log            # Error log
#SBATCH --partition=mcml-hgx-h100-94x4  # Partition name
#SBATCH --ntasks=2                      # Number of tasks
#SBATCH --qos=mcml                      # Quality of service
#SBATCH --gres=gpu:1                    # Number of GPUs (if needed)
#SBATCH --container-image="nvcr.io/nvidia/pytorch:24.11-py3"
#SBATCH -D ./
#SBATCH -t 2-00:00:00

cd ~/workspace/AmortizedCausalDiscovery/codebase
pwd

# ACD baseline
python -m train --suffix _springs5_dynamic1 --dynamic 1 --epochs=500 --encoder=mlp
python -m train --suffix _springs5_dynamic2 --dynamic 2 --epochs=500 --encoder=mlp
python -m train --suffix _springs5_dynamic3 --dynamic 3 --epochs=500 --encoder=mlp

# main transformer
python -m train --suffix _springs5_dynamic3 --dynamic 1 --epochs=500 --encoder=transformer_old
python -m train --suffix _springs5_dynamic2 --dynamic 2 --epochs=500 --encoder=transformer_old
python -m train --suffix _springs5_dynamic1 --dynamic 3 --epochs=500 --encoder=transformer_old

# recurrent transformer
python -m train --suffix _springs5_dynamic2 --dynamic 1 --epochs=500 --encoder=transformer
python -m train --suffix _springs5_dynamic3 --dynamic 2 --epochs=500 --encoder=transformer
python -m train --suffix _springs5_dynamic1 --dynamic 3 --epochs=500 --encoder=transformer
