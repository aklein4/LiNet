
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import math


"""
This file is used for experimenting with the efficient lossy compression of gradients using transfer functions.
"""


N_COMPONENTS = 5
SEQ_LEN = 10
DELAY = 5

K_RANGE = (0, 10)
OMEGA_RANGE = (0, 2*math.pi)
PHI_RANGE = (0, 2*math.pi)
POLE_RANGE = (-1, 0)

LOSSES = []
def target_function(params, target):

    if params.ndim != 1:
        raise ValueError('params must be 1d')

    k = params[:N_COMPONENTS]
    # omega = params[N_COMPONENTS:2*N_COMPONENTS]
    omega = np.array([i for i in range(N_COMPONENTS)])
    phi = params[2*N_COMPONENTS:3*N_COMPONENTS]
    p = params[3*N_COMPONENTS:]

    total_loss = 0
    for t in range(len(target)):
        signal = sum(k * np.sin(omega*t/SEQ_LEN + phi) * np.exp(p * t/SEQ_LEN))
        total_loss += (1 + 10*SEQ_LEN*target[t]) + (target[t] - signal) ** 2
    
    LOSSES.append(total_loss)
    return total_loss

def main():

    target = np.zeros([SEQ_LEN])
    target[DELAY] = 1

    components = optimize.dual_annealing(
        target_function,
        [K_RANGE for _ in range(N_COMPONENTS)] +
            [OMEGA_RANGE for _ in range(N_COMPONENTS)] +
            [PHI_RANGE for _ in range(N_COMPONENTS)] +
            [POLE_RANGE for _ in range(N_COMPONENTS)],
        args=(target,), maxiter=250
    ).x

    k = components[:N_COMPONENTS]
    # omega = components[N_COMPONENTS:2*N_COMPONENTS]
    omega = np.array([i for i in range(N_COMPONENTS)])
    phi = components[2*N_COMPONENTS:3*N_COMPONENTS]
    p = components[3*N_COMPONENTS:]

    print(
        "gains:", k,
        "\nfrequencies:", omega,
        "\nphases:", phi,
        "\npoles:", p
    )

    response = []
    high_freq = []
    sf = 100
    iter = 3
    for t in range(SEQ_LEN*iter):
        response.append(sum(k * np.sin(omega*(t/SEQ_LEN) + phi) * np.exp(p * (t)/SEQ_LEN)))
    for t in range(SEQ_LEN*sf*iter):
        high_freq.append(sum(k * np.sin(omega*(t/sf)/SEQ_LEN + phi) * np.exp(p * (t/sf)/SEQ_LEN)))
    
    plt.plot(response)
    #plt.plot([i/sf for i in range(SEQ_LEN*sf*iter)], high_freq)
    plt.show()

    # plt.clf()
    # plt.plot(LOSSES)
    # plt.show()

if __name__ == '__main__':
    main()