
import linet
from train import DataLoader, train

import torch
import argparse
import matplotlib.pyplot as plt

import numpy as np
import numpy.polynomial as poly


"""
Example implementation of LiNet to classify between polynomial and sinusoidal time functions.
run main.py -h for arguments
"""


NUM_STEPS = 64 # number of time steps per data point
LAYER_SIZE = 16 # number of nodes per layer
NUM_HIDDEN = 4 # number of hidden layers

N_TRAIN = 64 # number of training points per type
N_VAL = 16 # number of validation points per type


def get_random_poly(device):
    """
    Get a randomly generated polynomial time series tensor with all values in [0, 1]
    """

    # get poly
    poly_tensor = torch.zeros((NUM_STEPS, LAYER_SIZE), device=device)
    p = poly.Polynomial(np.random.randint([-5*NUM_STEPS**2, -5*NUM_STEPS, -5], [5*NUM_STEPS**2, 5*NUM_STEPS, 5], size=(3,)))
    for t in range(NUM_STEPS):
        poly_tensor[t][0] = p(t)

    # normalize
    poly_tensor /= 5*NUM_STEPS**2
    poly_tensor -= min(0, torch.min(poly_tensor).item())
    poly_tensor /= max(1, torch.max(poly_tensor).item())

    return poly_tensor

def get_random_sin(device):
    """
    Get a randomly generated sinusoid time series tensor with all values in [0, 1]
    """
    sin_tensor = torch.zeros((NUM_STEPS, LAYER_SIZE), device=device)
    
    # get sin
    w = np.random.random() + 0.025
    phi = np.random.random() * 2*np.pi
    A = np.random.random() / 2
    
    for t in range(NUM_STEPS):
        sin_tensor[t][0] = A*np.sin(w*t + phi)

    # normalize
    sin_tensor -= min(0, torch.min(sin_tensor).item())
    
    return sin_tensor


def main(args):

    # useful for debugging
    if args.full_print:
        torch.set_printoptions(profile="full")

    # get the correct device
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device('cuda:0')
    
    # init the net
    net = linet.LiNet(LAYER_SIZE, NUM_HIDDEN, device=device, last_activates=False, act_func='elu')
    
    # hold data
    train_poly = []
    train_sin = []
    val_poly = []
    val_sin = []

    # get data
    for n in range(N_TRAIN):
        train_poly.append(get_random_poly(device))
        train_sin.append(get_random_sin(device))
    for n in range(N_VAL):
        val_poly.append(get_random_poly(device))
        val_sin.append(get_random_sin(device))

    # show polynomials on graph
    for poly in val_poly:
        plt.plot(range(poly.shape[0]), poly[:, 0].cpu())
    plt.title("Polynomial Validation Functions")
    plt.savefig("poly_val.png")

    # show sin on graph
    plt.clf()
    for sin in val_sin:
        plt.plot(range(sin.shape[0]), sin[:, 0].cpu())
    plt.title("Sinusoid Validation Functions")
    plt.savefig("sin_val.png")

    # turn data into dataloaders
    train_data = DataLoader(
        train_poly + train_sin,
        [0 for _ in range(len(train_poly))] + [1 for _ in range(len(train_sin))]
    )
    val_data = DataLoader(
        val_poly + val_sin,
        [0 for _ in range(len(val_poly))] + [1 for _ in range(len(val_sin))]
    )

    # train with params
    train(
        net, training_data=train_data, validation_data=val_data,
        learning_rate=1e-4, batch_size=1, plot=True, checkpoint_freq=None,
        shuffle=True, classifier=True
    )
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('--full', dest='full_print', action='store_const', const=True, default=False, 
                    help='Whether to print entire tensors')

    args = parser.parse_args()
    main(args)