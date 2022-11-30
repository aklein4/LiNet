
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import math

"""
This file is used for experimenting with the efficient lossy compression of gradients using fourier transforms.
"""


N_COMPONENTS = 100
SEQ_LEN = 1000
DELAY = SEQ_LEN//2
SF = math.pi/SEQ_LEN
WIDTH = 0.01

def R(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]], dtype=np.float32)

def response_calc(t_elapsed, n_params, delay, width=WIDTH):
    maxi = width/2
    for i in range(1, n_params):
        a = math.sin(width*i*math.pi)/(i*math.pi)
        if width == 0:
            a = 1
        maxi +=  a

    total = width/2
    for i in range(1, n_params):
        a = math.sin(width*i*math.pi)/(i*math.pi)
        if width == 0:
            a = 1
        total +=  a * math.cos(math.pi * i * (t_elapsed-delay)) / maxi
    return total

def delay_conv(sequence, n_params, width=WIDTH):
    maxi = 0
    for t in range(len(sequence)):
        for i in range(1, n_params):
            a = math.sin(width*i*math.pi)/(i*math.pi)
            if width == 0:
                a = 1
            maxi +=  a * math.sin(t*i*math.pi/len(sequence))

    a = [(width/2)/maxi]
    for i in range(1, n_params):
        if width == 0:
            a.append(1/maxi)
        else:
            a.append(math.sin(width*i*math.pi)/(i*math.pi)/maxi)

    signal = []
    convs = [np.array([[0], [0]], dtype=np.float32) for p in range(n_params)]
    for t in range(math.ceil(len(sequence)*2.5)):

        if t < len(sequence):
            for i in range(len(convs)):
                convs[i] += np.matmul(R(i * math.pi), np.array([[sequence[t]], [0]], dtype=np.float32))

        signal.append(sum([a[i] * convs[i][0] for i in range(len(convs))]))

        for i in range(len(convs)):
            convs[i] = np.matmul(R(i*math.pi/len(sequence)), convs[i])

    return signal

def main():

    sequence = [0]
    for i in range(1, SEQ_LEN):
        #sequence.append(0)
        sequence.append(.99*sequence[-1] + (1-.99)*np.random.normal(scale=.1))

    signal = delay_conv(sequence, N_COMPONENTS, WIDTH)

    avg = sum([abs(s) for s in sequence])/len(sequence)
    error = 0
    for i in range(len(sequence)):
        error += abs(sequence[i] - signal[i+len(sequence)])/avg
    print(error/len(sequence))

    plt.plot(sequence)
    plt.plot(signal)
    #plt.plot([[responses[i][t] for i in range(len(responses))] for t in range(len(responses[0]))])
    plt.show()

if __name__ == '__main__':
    main()