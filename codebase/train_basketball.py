#!/usr/bin/env python3

"""
Training script for basketball player trajectory data.
This script trains the amortized causal discovery model on basketball data instead of springs data.
"""

from __future__ import division
from __future__ import print_function

import sys
import os
from collections import defaultdict
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the codebase directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.modules import *
from utils import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader

if __name__ == "__main__":
    plt.switch_backend('agg')  # Use non-interactive backend
    
    # Parse arguments but override basketball-specific ones
    args = arg_parser.parse_args()
    
    # Override key parameters for basketball data
    args.suffix = "_basketball10"  # 10 players
    args.num_atoms = 10  # 10 players total (5 per team)
    args.timesteps = 730  # Use the minimum timesteps from our data
    args.dims = 4  # position (x,y) + velocity (vx,vy)
    
    # Adjust other parameters as needed
    if not hasattr(args, 'datadir') or args.datadir == "./data":
        args.datadir = "data"
    
    # Set reasonable defaults for basketball
    if args.lr == 0.0005:  # default learning rate
        args.lr = 0.001
    if args.epochs == 500:  # if using default
        args.epochs = 200  # Fewer epochs for initial testing
    
    print(f"Training basketball model with:")
    print(f"  - Suffix: {args.suffix}")
    print(f"  - Num atoms (players): {args.num_atoms}")
    print(f"  - Timesteps: {args.timesteps}")
    print(f"  - Dimensions: {args.dims}")
    print(f"  - Data directory: {args.datadir}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    
    # Initialize logger
    logs = logger.Logger(args)

    if args.GPU_to_use is not None:
        logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

    # Load basketball data
    (
        train_loader,
        valid_loader,
        test_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    ) = data_loader.load_data(args)

    rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)

    encoder, decoder, optimizer, scheduler, edge_probs = model_loader.load_model(
        args, loc_max, loc_min, vel_max, vel_min
    )

    logs.write_to_log_file(encoder)
    logs.write_to_log_file(decoder)

    if args.prior != 1:
        assert 0 <= args.prior <= 1, "args.prior not in the right range"
        prior = np.array(
            [args.prior]
            + [
                (1 - args.prior) / (args.edge_types - 1)
                for _ in range(args.edge_types - 1)
            ]
        )
        logs.write_to_log_file("Using prior")
        logs.write_to_log_file(prior)
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = log_prior.unsqueeze(0).unsqueeze(0)

        if args.cuda:
            log_prior = log_prior.cuda()
    else:
        log_prior = None

    if args.global_temp:
        args.categorical_temperature_prior = utils.get_categorical_temperature_prior(
            args.alpha, args.num_cats, to_cuda=args.cuda
        )

    # Train model
    def train():
        best_val_loss = np.inf
        best_epoch = 0
        reg_weight = 4000

        for epoch in range(args.epochs):
            t_epoch = time.time()
            train_losses = defaultdict(list)

            for batch_idx, minibatch in enumerate(train_loader):

                data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

                optimizer.zero_grad()

                losses, _, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
                    args,
                    encoder,
                    decoder,
                    data,
                    relations,
                    rel_rec,
                    rel_send,
                    args.hard,
                    edge_probs=edge_probs,
                    log_prior=log_prior,
                    temperatures=temperatures,
                    reg_weight=reg_weight
                )

                loss = losses["loss"]

                loss.backward()
                optimizer.step()

                train_losses = utils.append_losses(train_losses, losses)

            string = logs.result_string("train", epoch, train_losses, t=t_epoch)
            logs.write_to_log_file(string)
            logs.append_train_loss(train_losses)
            scheduler.step()

            # Validation
            if args.validate:
                val_losses = defaultdict(list)
                encoder.eval()
                decoder.eval()
                for batch_idx, minibatch in enumerate(valid_loader):
                    data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

                    with torch.no_grad():
                        losses, _, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
                            args,
                            encoder,
                            decoder,
                            data,
                            relations,
                            rel_rec,
                            rel_send,
                            args.hard,
                            edge_probs=edge_probs,
                            log_prior=log_prior,
                            temperatures=temperatures,
                            reg_weight=reg_weight
                        )

                    val_losses = utils.append_losses(val_losses, losses)

                string = logs.result_string("val", epoch, val_losses)
                logs.write_to_log_file(string)
                logs.append_val_loss(val_losses)

                if val_losses["loss"][-1] < best_val_loss:
                    best_val_loss = val_losses["loss"][-1]
                    best_epoch = epoch

                encoder.train()
                decoder.train()

        return best_epoch, epoch

    if args.epochs != 0:
        try:
            if args.test_time_adapt:
                raise KeyboardInterrupt

            best_epoch, epoch = train()

        except KeyboardInterrupt:
            best_epoch, epoch = -1, -1
    else:
        best_epoch, epoch = -1, -1

    print("Optimization Finished!")
    logs.write_to_log_file("Best Epoch: {:04d}".format(best_epoch)) 