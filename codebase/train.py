from __future__ import division
from __future__ import print_function

from collections import defaultdict

import os
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

    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_losses = defaultdict(list)

        for batch_idx, minibatch in enumerate(train_loader):

            data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

            optimizer.zero_grad()

            losses, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
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
            losses, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
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

            losses, output, unobserved, _, = forward_pass_and_eval.forward_pass_and_eval(
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

            for j in range(1):
                ############ Plotting ############
                import os
                import matplotlib.pyplot as plt

                # print(data.shape)
                # print(output.shape)
                # #print(unobserved.shape) # 0 if None, else Tensor
                # print(data_decoder.shape)
                # print("-------------------")

                plot_dir = os.path.join(logs.args.log_path, "plots")
                os.makedirs(plot_dir, exist_ok=True)

                # data shape = [batch_size, #particles, trajectory_length, 4]
                # We are only using the first two coordinates: [:, :2]
                plot_data = data[j, :, :, :2].cpu()     # shape: (#particles, trajectory_length, 2)
                plot_output = output[j, :, :, :2].cpu() # shape: (#particles, trajectory_length, 2)

                plt.figure()
                colors = ["C0", "C1", "C2", "C3", "C4"]
                
                has_unobserved = isinstance(unobserved, torch.Tensor) or data.shape[1] != output.shape[1]
                has_unobserved_prediction = has_unobserved and data.shape[1] == output.shape[1]
                #print(has_unobserved)
                #print(has_unobserved_prediction)

                # 1) Plot the data (with alpha=0.5 and a filled circle at the end)
                num_observed_particles = plot_data.shape[0] if not has_unobserved else plot_data.shape[0] - 1
                for i in range(num_observed_particles):  # Loop over particles
                    x = plot_data[i, :, 0].numpy()
                    y = plot_data[i, :, 1].numpy()

                    # Plot the trajectory with alpha=0.5
                    plt.plot(x, y, alpha=0.5, linewidth=1)

                    # Plot a filled circle at the last point of the trajectory
                    plt.plot(x[-1], y[-1], marker='o', markersize=8, color=colors[i], alpha=0.5)

                # 2) Plot the output (with alpha=1.0), using the same color as the corresponding data
                for i in range(num_observed_particles):
                    x_out = plot_output[i, :, 0].numpy()
                    y_out = plot_output[i, :, 1].numpy()

                    plt.plot(x_out, y_out, alpha=1.0, color=colors[i], linewidth=2)
                    plt.plot(x_out[-1], y_out[-1], marker='o', markersize=8, color=colors[i], alpha=1.0)

                # Save the main plot
                xlim = plt.xlim()
                ylim = plt.ylim()
                plt.savefig(os.path.join(plot_dir, f"data_{batch_idx:03}_{j:03}.png"))
                plt.close()

                ##############################################
                # Only plot the *last* particle to a separate
                # plot if data_decoder.shape[0] != data.shape[0]
                ##############################################
                if has_unobserved:
                    # Create a new figure for the last particle
                    plt.figure()

                    # Extract the last particle from ground truth and output
                    last_particle_data = plot_data[-1]   # shape: (trajectory_length, 2)

                    # Plot true trajectory (with alpha=0.5)
                    color = colors[-1]
                    x_true = last_particle_data[:, 0].numpy()
                    y_true = last_particle_data[:, 1].numpy()
                    plt.plot(x_true, y_true, alpha=0.5, color=color, linewidth=1)
                    plt.plot(x_true[-1], y_true[-1], marker='o', markersize=8, color=color, alpha=0.5)

                    if has_unobserved_prediction:
                        last_particle_out  = plot_output[-1]
                        last_particle_out = unobserved[j, 0, :, :2].cpu()

                        x_out = last_particle_out[:, 0].numpy()
                        y_out = last_particle_out[:, 1].numpy()

                        plt.plot(x_out, y_out, alpha=1.0, color=color, linewidth=2)
                        plt.plot(x_out[-1], y_out[-1], marker='o', markersize=8, color=color, alpha=1.0)

                    # Save this separate figure
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.savefig(os.path.join(plot_dir, f"data_{batch_idx:03}_{j:03}_unobserved.png"))
                    plt.close()



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
    try:
        if args.test_time_adapt:
            raise KeyboardInterrupt

        best_epoch, epoch = train()

    except KeyboardInterrupt:
        best_epoch, epoch = -1, -1

    print("Optimization Finished!")
    logs.write_to_log_file("Best Epoch: {:04d}".format(best_epoch))

    if args.test:
        test(encoder, decoder, epoch)
