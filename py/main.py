
import ordinet

import torch
import argparse
import matplotlib.pyplot as plt
import time

NUM_STEPS = 60
LAYER_SIZE = 1024
NUM_RECORD = 16
NUM_HIDDEN = 8

def main(args):
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device('cuda:0')
    net = ordinet.Ordinet(LAYER_SIZE, NUM_HIDDEN, device=device)

    outputs = []

    x_on = torch.full([net.layer_size], 1.0, device=device)
    x_off = torch.zeros([net.layer_size], device=device)

    start_time = time.time_ns()
    for i in range(NUM_STEPS):
        y = None
        if i == 0:
            y = net.forward(x_on)
        else:
            y = net.forward(x_off)
        
        outputs.append(y.tolist()[:NUM_RECORD])
    print("Time:", round((time.time_ns()-start_time)/1000000, 2))

    plt.plot([[i for _ in range(NUM_RECORD)] for i in range(NUM_STEPS)], outputs)
    plt.savefig('output.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)