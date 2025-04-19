from __future__ import division
from __future__ import print_function

from collections import defaultdict

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from model.modules import *
from utils import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader


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

        if args.validate:
            val_losses = val(epoch)
            val_loss = np.mean(val_losses["loss"])
            if val_loss < best_val_loss:
                print("Best model so far, saving...")
                logs.create_log(
                    args,
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    accuracy=np.mean(val_losses["acc"]),
                )
                best_val_loss = val_loss
                best_epoch = epoch
        elif (epoch + 1) % 100 == 0:
            logs.create_log(
                args,
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                accuracy=np.mean(train_losses["acc"]),
            )

        logs.draw_loss_curves()

    return best_epoch, epoch


def val(epoch):
    t_val = time.time()
    val_losses = defaultdict(list)

    if args.use_encoder:
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
                True,
                edge_probs=edge_probs,
                log_prior=log_prior,
                testing=True,
                temperatures=temperatures,
            )

        val_losses = utils.append_losses(val_losses, losses)

    string = logs.result_string("validate", epoch, val_losses, t=t_val)
    logs.write_to_log_file(string)
    logs.append_val_loss(val_losses)

    if args.use_encoder:
        encoder.train()
    decoder.train()

    return val_losses


def test(encoder, decoder, epoch):
    args.shuffle_unobserved = False
    # args.prediction_steps = 49
    test_losses = defaultdict(list)

    if args.load_folder == "":
        ## load model that had the best validation performance during training
        if args.use_encoder:
            encoder.load_state_dict(torch.load(args.encoder_file))
        decoder.load_state_dict(torch.load(args.decoder_file))

    if args.use_encoder:
        encoder.eval()
    decoder.eval()

    for batch_idx, minibatch in enumerate(test_loader):

        data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

        with torch.no_grad():
            assert (data.size(2) - args.timesteps) >= args.timesteps

            data_encoder = data[:, :, : args.timesteps, :].contiguous()
            data_decoder = data[:, :, args.timesteps : -1, :].contiguous()
            relations = relations[:, :args.timesteps].contiguous()

            losses, _, _, edges, factors = forward_pass_and_eval.forward_pass_and_eval(
                args,
                encoder,
                decoder,
                data,
                relations,
                rel_rec,
                rel_send,
                True,
                data_encoder=data_encoder,
                data_decoder=data_decoder,
                edge_probs=edge_probs,
                log_prior=log_prior,
                testing=True,
                temperatures=temperatures,
            )

            if batch_idx < 2:
                ground_truth_edges = relations
                predicted_edges = torch.argmax(edges, dim=-1)[:, :ground_truth_edges.size(1)]

                batch_size = ground_truth_edges.size(0)
                for sample_index in range(batch_size):
                    # Convert tensor to numpy arrays (detach if needed)
                    sample_ground_truth_edges = ground_truth_edges[sample_index].detach().cpu().numpy()
                    sample_predicted_edges = predicted_edges[sample_index].detach().cpu().numpy()

                    # Calculate the difference map
                    # 0 = correct prediction
                    # 1 = false positive (predicted edge not in GT)
                    # 2 = false negative (GT edge not predicted)
                    difference_map = np.zeros_like(sample_ground_truth_edges)
                    difference_map[(sample_predicted_edges == 1) & (sample_ground_truth_edges == 0)] = 1  # FP
                    difference_map[(sample_predicted_edges == 0) & (sample_ground_truth_edges == 1)] = 2  # FN

                    # Create custom colormap for error map
                    from matplotlib.colors import ListedColormap
                    cmap_diff = ListedColormap(['black', 'red', 'blue'])  # 0: correct, 1: FP, 2: FN

                    # Create a figure with three subplots
                    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

                    # Determine dimensions: rows (height) and columns (width) of the images
                    num_rows, num_cols = sample_ground_truth_edges.shape

                    # Helper function to add grid lines at every pixel boundary
                    def add_pixel_grid(ax, num_rows, num_cols):
                        # Draw horizontal grid lines
                        for y in range(1, num_rows):
                            ax.axhline(y - 0.5, color='gray', linestyle='-', linewidth=0.8)
                        # Draw vertical grid lines
                        for x in range(1, num_cols):
                            ax.axvline(x - 0.5, color='gray', linestyle='-', linewidth=0.2)

                    # Ground Truth subplot
                    axes[0].imshow(1 - sample_ground_truth_edges, cmap='gray', interpolation='nearest')
                    add_pixel_grid(axes[0], num_rows, num_cols)
                    axes[0].set_title("Ground Truth", fontsize=20)
                    axes[0].axis('off')  # Removes the axis labels and ticks

                    # Predicted subplot
                    axes[1].imshow(1 - sample_predicted_edges, cmap='gray', interpolation='nearest')
                    add_pixel_grid(axes[1], num_rows, num_cols)
                    axes[1].set_title("Predicted", fontsize=20)
                    axes[1].axis('off')

                    # Error Map subplot
                    axes[2].imshow(difference_map, cmap=cmap_diff, interpolation='nearest')
                    add_pixel_grid(axes[2], num_rows, num_cols)
                    axes[2].set_title("Error Map", fontsize=20)
                    axes[2].axis('off')

                    # Save the output
                    plt.tight_layout()
                    import os
                    out_path = os.path.join(args.plotdir, f"sample_{batch_idx}_{sample_index}_grid_with_diff.png")
                    plt.savefig(out_path)
                    plt.close(fig)


                    # Optionally print the arrays to console
                    #print("Ground truth edges:")
                    ##print(sample_ground_truth_edges)
                    #print("Predicted edges:")
                    ##print(sample_predicted_edges)
                    #print("Factors:")
                    ##print(factors[sample_index])
                    #print("-----------------")
            

        test_losses = utils.append_losses(test_losses, losses)

    string = logs.result_string("test", epoch, test_losses)
    logs.write_to_log_file(string)
    logs.append_test_loss(test_losses)

    logs.create_log(
        args,
        decoder=decoder,
        encoder=encoder,
        optimizer=optimizer,
        final_test=True,
        test_losses=test_losses,
    )


if __name__ == "__main__":
    plt.switch_backend('agg')

    args = arg_parser.parse_args()
    logs = logger.Logger(args)

    if args.GPU_to_use is not None:
        logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

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

    ##Train model
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

    if args.test:
        test(encoder, decoder, epoch)
