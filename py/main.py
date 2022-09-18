
import ordinet

import torch
import argparse
import matplotlib.pyplot as plt
import time

NUM_STEPS = 20
LAYER_SIZE = 4
NUM_RECORD = 16
NUM_HIDDEN = 2

def main(args):

    # useful for debugging
    if args.full_print:
        torch.set_printoptions(profile="full")

    # get the correct device
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device('cuda:0')
    
    # init the net
    net = ordinet.Ordinet(LAYER_SIZE, NUM_HIDDEN, device=device, last_activates=False)

    inputs = []
    outputs = []
    loss_outputs = []

    x_on = torch.full([net.layer_size], 1.0, device=device)
    x_off = torch.zeros([net.layer_size], device=device)

    start_time = time.time_ns()
    for i in range(NUM_STEPS):
        y = None
        if i == 0:
            y = net.forward(x_on)
            inputs.append(x_on)
        else:
            y = net.forward(x_off)
            inputs.append(x_off)
        
        outputs.append(y.tolist()[:min(NUM_RECORD, LAYER_SIZE)])
        loss_outputs.append(-y)
    print("Forward Time:", round((time.time_ns()-start_time)/1000000, 2))

    input_seq = torch.stack(inputs)
    output_seq = torch.stack(loss_outputs)
    
    start_time = time.time_ns()
    net.backward(input_seq, output_seq)
    print("Backward Time:", round((time.time_ns()-start_time)/1000000, 2))

    print(net.gain_grads)

    plt.plot([[i for _ in range(min(NUM_RECORD, LAYER_SIZE))] for i in range(NUM_STEPS)], outputs)
    plt.savefig('output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('--full', dest='full_print', action='store_const', const=True, default=False, 
                    help='Whether to print entire tensors')

    args = parser.parse_args()
    main(args)