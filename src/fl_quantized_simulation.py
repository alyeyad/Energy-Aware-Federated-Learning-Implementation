#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

from low_precision_quantizer import *

if __name__ == '__main__':
    start_time = time.time()
    bandwidth_mbps = 10
    # define paths
    path_project = os.path.abspath('..')
    os.makedirs('logs', exist_ok=True)
    logger = SummaryWriter('logs')
    logger.add_scalar('ss', 9)

    args = args_parser()
    exp_details(args)

    if args.gpu and int(args.gpu):
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu and int(args.gpu) else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    num_levels = 64

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    savings = []
    delays = []
    for epoch in tqdm(range(args.epochs)):
        # init local weights and loss
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        # sample a fraction of users (with args frac)
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        average_savings = dict()

        # for each sampled user
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            print(f"User Index: {idx}")
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            quantized_w_bits = dict()
            quantized_w = dict()
            savings_info = dict()
            level_mappings = dict()
            delays_info = dict()
            for k in w.keys():
                quantized_matrix, savings_info, quantized_matrix_bits, level_mapping, sign_vector = low_precision_quantizer_4d(
                    w[k], num_levels)
                delay_info = simulate_delay_4d(quantized_matrix, num_levels, bandwidth_mbps)
                quantized_w_bits[k] = quantized_matrix_bits
                quantized_w[k] = quantized_matrix
                savings_info[k] = savings_info
                level_mappings[k] = level_mapping
                delays_info[k] = delay_info

            savings.append(savings_info)
            delays.append(delays_info)

            # delay_info = simulate_delay(original_vector, num_levels, bandwidth_mbps, sleep_for_delay=True)

            # reconstructed_weights = bit_vector_to_weights(quantized_vector_bits, level_mapping)
            local_weights.append(copy.deepcopy(quantized_w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_quantized_{}.pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, num_levels)

    total_time = time.time() - start_time
    with open(file_name, 'wb') as f:
        pickle.dump({"train_loss": train_loss,
                     "train_accuracy": train_accuracy,
                     "total_time": total_time,
                     "test_accuracy": test_acc,
                     "C": args.frac,
                     "epochs": args.epochs,
                     "B": args.local_bs,
                     "iid": args.iid,
                     "E": args.local_ep,
                     "delays": delays,
                     "savings": savings}, f)

    print('\n Total Run Time: {0:0.4f}'.format(total_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_quantized_{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, num_levels))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_quantized_{}_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, num_levels))
