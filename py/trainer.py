
import ordinet

import torch
import argparse
import matplotlib.pyplot as plt
import time
import math
import sys
import csv

NUM_STEPS = 20
LAYER_SIZE = 64
NUM_RECORD = 1
NUM_HIDDEN = 2

def read_data(file, keys, padded_size, flip=False):
    data = []
    index_loc = []
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, dialect='excel')
        header = True
        for row in spamreader:
            if header:
                header = False
                for key in keys:
                    if key not in row:
                        raise ValueError("Header key not found.")
                    index_loc.append(row.index(key))
            else:
                data_entry = torch.zeros([padded_size])
                for i in range(len(index_loc)):
                    data_entry[i] = float(row[index_loc[i]])
                data.append(data_entry)
    if flip:
        data.reverse()
    return torch.stack(data)


def main(args):

    # useful for debugging
    if args.full_print:
        torch.set_printoptions(profile="full")

    # get the correct device
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device('cuda:0')

    # init the net
    net = ordinet.Ordinet(LAYER_SIZE, NUM_HIDDEN, device=device, last_activates=False, act_func='relu')

    # get the training data
    data = read_data('training_data/BTC-USD.csv', ["High", "Low"], 10, flip=True)

    # format data to match the net
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('--full', dest='full_print', action='store_const', const=True, default=False, 
                    help='Whether to print entire tensors')

    args = parser.parse_args()
    main(args)