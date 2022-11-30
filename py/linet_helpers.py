
import torch
import time

"""
Helper constants and functions for LiNet and related functions.
"""


class Timer:
    # a simple timer class to do my bidding
    def __init__(self):
        self.start_time = time.time_ns()
    def reset(self):
        self.start_time = time.time_ns()
    def get_time(self):
        return (time.time_ns() - self.start_time)/1000000
    def print(self):
        print((time.time_ns() - self.start_time)/1000000)

        
ACT_FUNCS = {
    # map string to activation functions
    "identity": torch.nn.Identity(),
    "relu": torch.nn.ReLU(inplace=False),
    "elu": torch.nn.ELU(inplace=False),
    "sigmoid": torch.nn.Sigmoid()
}

SIG_FUNC = torch.nn.Sigmoid()
DIFF_FUNCS = {
    # map string to derivative of activation functions
    "identity": lambda t: torch.full_like(t, 1),
    "relu": lambda t: torch.where(t.gt(0), 1, 0),
    "elu": lambda t: torch.where(t.ge(0), 1, torch.exp(t)),
    "sigmoid": lambda t: SIG_FUNC(t) * (1 - SIG_FUNC(t))
}


def indw(A, i):
    # evaluate A at wrapped index i
    if i < 0:
        return A[A.size(dim=0) - i]
    if i >= A.size(dim=0):
        return A[i % A.size(dim=0)]
    return A[i]

def indw_i(A, i):
    # get wrapped index i
    if i < 0:
        return A.size(dim=0) - i
    elif i >= A.size(dim=0):
        return i % A.size(dim=0)
    return i

def indp(A, i):
    # get 0-padded value of A at i
    if i >= 0 and i < A.size(dim=0):
        return A[i]
    return A[0] * 0