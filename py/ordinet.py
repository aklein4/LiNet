
from unittest.mock import DEFAULT
import numpy as np
import math
import torch

ACT_FUNCS = {
    "relu": torch.nn.ReLU(inplace=True)
}

INTEGRATOR_LIMIT = -0.05
DEFAULT_MAX_TAU = 6

class Ordinet:

    def __init__(self, layer_size, num_hidden, device=torch.device("cpu"), dtype=torch.float32, act_func='ReLU', last_activates=True):

        # TODO: Bias activation in every layer

        # type and location of data
        self.device = device
        self.dtype = dtype

        # information about the size of the net
        self.layer_size = layer_size
        self.num_layers = num_hidden + 2
        self.container_size = (self.num_layers-1, self.layer_size, self.layer_size)

        # tensors to do the heavy computational lifting
        self.activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)
        self.gammas = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)
        self.gains = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

        # initialize the tensors with good starting weights
        for b in range(self.container_size[0]):
            self._init_layer(self.gains, self.gammas, b)

        # keep track of the gradients for the weights to be applied
        self.gain_grads = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)
        self.pole_grads = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

        # the activation function between on the summation
        try:
            self.act_func = ACT_FUNCS[act_func.lower()]
        except:
            raise("Error: invalid activation function name.")
        # whether the last activation should have an activation function
        self.last_activates = last_activates # TODO: Put this in backward pass


    def _init_layer(self, gains, gammas, index):
        # mutate gain and gamma values

        if gains[index].size() != gammas[index].size():
            raise("Error: Layer initialization has gain and gamma tensor sizes that do not match.")
        if gains[index].dim() != 2:
            raise("Error: layer initialization should be done on an indexed 2D layer.")
        n_inputs = gains[index].size(dim=1)
        matrix_size = gains[index].size(dim=0), n_inputs

        # He initialization on integral and gain
        integrals = torch.normal(0.0, math.sqrt(2/n_inputs), matrix_size, dtype=self.dtype, device=self.device)
        gains[index] = torch.normal(0.0, math.sqrt(2/n_inputs), matrix_size, dtype=self.dtype, device=self.device)

        # select pole to give the integral
        poles = torch.div(gains[index], integrals)
        poles = torch.abs(poles)
        poles *= -1

        # make sure none of the poles cross the integrator limit
        poles = torch.minimum(poles, torch.full(matrix_size, INTEGRATOR_LIMIT, dtype=self.dtype, device=self.device))

        # convert the poles to gammas
        gammas[index] = torch.exp(poles)

    def forward(self, phi: torch.Tensor):
        if phi.dim() != 1:
            raise("Error: Input tensor to forward must be 1 dimensional.")
        if phi.size(dim=0) != self.layer_size:
            raise("Error: Input tensor to forward is the wrong size.")
        if phi.device != self.device:
            raise("Error: Input tensor to forward does not match device with Ordinet.")
        if phi.dtype != self.dtype:
            raise("Error: Input tensor to forwrd does not math dtype with Ordinet.")

        y = phi.clone()

        for b in range(self.container_size[0]):
            self.activations[b] = torch.mul(self.activations[b], self.gammas[b])
            self.activations[b] = torch.add(self.activations[b], torch.mul(self.gains[b], torch.t(y)))

            y = torch.sum(self.activations[b], dim=1)
            if b != self.container_size[0]-1 or self.last_activates:
                y = self.act_func(y)

        return y

    def reset(self):
        self.activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)


    def backward(self, x_sequence, loss_gradients, max_tau=DEFAULT_MAX_TAU, d_offset = 0.99):
        if x_sequence.size(dim=0) != loss_gradients.size(dim=0):
            raise("Error: x_sequence and loss_gradients sizes do not match on temporal dimension")
        if x_sequence.size(dim=1) != self.layer_size:
            raise("Error: x_sequence has wrong input size.")
        if loss_gradients.size(dim=1) != self.layer_size:
            raise("Error: loss_gradients has wrong output size.")
        if x_sequence.device != self.device:
            raise("Error: x_sequence tensor does not match device with Ordinet.")
        if loss_gradients.device != self.device:
            raise("Error: loss_gradients tensor does not match device with Ordinet.")
        if x_sequence.dtype != self.dtype:
            raise("Error: x_sequence tensor does not match device with Ordinet.")
        if loss_gradients.dtype != self.dtype:
            raise("Error: loss_gradients tensor does not match device with Ordinet.")

        # the length of the sequence we are training on
        sequence_length = x_sequence.size(dim=0)

        # initialize the leading net activations
        leading_activations = torch.zeros((self.container_size[0]-1, self.container_size[1], self.container_size[2]),
                                            dtype=self.dtype, device=self.device) # leading = trailing for last layer
        # initialize gradient-calculating per-transfer values
        trailing_activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)
        offset_activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

        # keep track of the backwards loss dependencies
        loss_convolutions = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

        # calculate the poles for each gamma
        poles = torch.log(self.gammas)
        # find the maximum forward-search depth
        max_depth = math.floor((-1/INTEGRATOR_LIMIT) * max_tau)
        # find the forward-search depth of each transfer
        search_depths = torch.floor(max_tau*torch.div(torch.full(self.container_size, 1.0, dtype=self.dtype, device=self.device), -1*poles))

        # contain the layer-wise inputs generated by the leading activations
        input_trails = [None for _ in range(self.num_layers-3)]
        for l in range(self.num_layers-3):
            input_trails[l] = torch.zeros([max_depth*(self.num_layers-3-l), self.layer_size], dtype=self.dtype, device=self.device)
        # point to where we are in the trail
        input_trail_indexes = [0 for _ in range(self.num_layers-3)]

        # contain the layer-wise loss gradients along the bottom staircase
        loss_trails  = [None for _ in range(self.num_layers-3)]
        for l in range(self.num_layers-3):
            loss_trails[l] = torch.zeros([max_depth, self.layer_size], dtype=self.dtype, device=self.device)
        # point to where we are in the trail
        loss_trail_indexes = [0 for _ in range(self.num_layers-3)]

        # TODO: last layer must accumulate loss before leading edge begins moving forward

        # iterate forward through the dataset
        for t in range(sequence_length + (self.num_layers-3)*max_depth ):
            
            if t < sequence_length:
                # do the leading forward pass
                input_leader = x_sequence[t].clone()
                for b in range(self.container_size[0]-1):
                    leading_activations[b] = torch.mul(leading_activations[b], self.gammas[b])
                    leading_activations[b] = torch.add(leading_activations[b], torch.mul(self.gains[b], torch.t(input_leader)))

                    input_leader = torch.sum(leading_activations[b], dim=1)
                    input_leader = self.act_func(input_leader)

                    # if this layer must store, then store
                    if b < self.num_layers-3:
                        input_trails[input_trail_indexes[b]] = input_leader.clone()
            
                # this input leader can be used for the final transfer input

                