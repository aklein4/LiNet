
import ordinet

import torch
import argparse
import matplotlib.pyplot as plt
import time
import math
import sys

NUM_STEPS = 20
LAYER_SIZE = 64
NUM_RECORD = 1
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
    net = ordinet.Ordinet(LAYER_SIZE, NUM_HIDDEN, device=device, last_activates=False, act_func='relu')

    x_on = torch.full([net.layer_size], 1.0, device=device)
    x_off = torch.zeros([net.layer_size], device=device)

    target = []
    for i in range(NUM_STEPS):
        target.append([math.sin(i * math.pi/20)])

    for epoch in range(10000):
        inputs = []
        outputs = []
        loss_outputs = []

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
            loss_grad = torch.zeros_like(y)
            loss_grad[0] = (math.sin(i * math.pi/20) - y[0])
            loss_outputs.append(loss_grad)
        net.reset()
        #print("Forward Time:", round((time.time_ns()-start_time)/1000000, 2))

        input_seq = torch.stack(inputs)
        output_seq = torch.stack(loss_outputs)
        
        start_time = time.time_ns()
        net.backward(input_seq, output_seq)
        #print("Backward Time:", round((time.time_ns()-start_time)/1000000, 2))

        loss = torch.sum(torch.square(output_seq)).item()
        if epoch % 2 == 0:
            print("Loss:", loss)
            plt.cla()
            plt.plot([[i for _ in range(min(NUM_RECORD, LAYER_SIZE))] for i in range(NUM_STEPS)], outputs)
            plt.plot([[i for _ in range(min(NUM_RECORD, LAYER_SIZE))] for i in range(NUM_STEPS)], target, 'k--')
            plt.savefig('output.png')

        net.apply_grads(0.001)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('--full', dest='full_print', action='store_const', const=True, default=False, 
                    help='Whether to print entire tensors')

    args = parser.parse_args()
    main(args)