#!/bin/bash

# transformer with dynamic graph
python train.py --suffix _basketball10 --num_atoms 10 --timesteps 700 --dims 4 --epochs 32 --lr 1e-3 --batch_size 4 --save_folder logs_basketball_transformer --prediction_steps 20 --encoder transformer_old
# mlp with static graph fromt the ACD paper
python train.py --suffix _basketball10 --num_atoms 10 --timesteps 700 --dims 4 --epochs 32 --lr 1e-3 --batch_size 4 --save_folder logs_basketball_transformer --prediction_steps 20 --encoder mlp