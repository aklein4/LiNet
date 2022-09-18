
import numpy as np
import math
import torch

import time
import sys

class Timer:
    def __init__(self):
        self.start_time = time.time_ns()
    def reset(self):
        self.start_time = time.time_ns()
    def get_time(self):
        return (time.time_ns() - self.start_time)/1000000
    def print(self):
        print((time.time_ns() - self.start_time)/1000000)

ACT_FUNCS = {
    "relu": torch.nn.ReLU(inplace=True)
}
DIFF_FUNCS = {
    "relu": lambda t: torch.where(t.gt(0), 1, 0)
}

MAX_SETTLING_TIME = 15
DEFAULT_MAX_TAU = 6
INTEGRATOR_LIMIT = -(1/(MAX_SETTLING_TIME/DEFAULT_MAX_TAU))

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

class Ordinet:

    def __init__(self, layer_size, num_hidden, device=torch.device("cpu"), dtype=torch.float32, act_func='ReLU', last_activates=True):

        # TODO: Bias activation in every layer

        # type and location of data
        self.device = device
        self.dtype = dtype

        # information about the size of the net
        self.layer_size = layer_size
        self.num_hidden = num_hidden
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
            raise ValueError("Invalid activation function name.")
        # the gradient of the activation function between on the summation
        try:
            self.act_grad_func = DIFF_FUNCS[act_func.lower()]
        except:
            raise ValueError("Invalid activation function name (diff key).")
        # whether the last activation should have an activation function
        self.last_activates = last_activates # TODO: Put this in backward pass


    def _init_layer(self, gains, gammas, index):
        # mutate gain and gamma values

        if gains[index].size() != gammas[index].size():
            raise ValueError("Layer initialization has gain and gamma tensor sizes that do not match.")
        if gains[index].dim() != 2:
            raise ValueError("Layer initialization should be done on an indexed 2D layer.")
        n_inputs = gains[index].size(dim=1)
        matrix_size = gains[index].size(dim=0), n_inputs

        # He initialization on integral and gain
        integrals = torch.normal(0.0, math.sqrt(2/n_inputs), matrix_size, dtype=self.dtype, device=self.device)
        gains[index] = torch.normal(0.0, math.sqrt(1/n_inputs), matrix_size, dtype=self.dtype, device=self.device) # 1 seems to work better for 60fps

        #integrals = torch.subtract(integrals, gains[index])

        # select pole to give the integral
        poles = torch.div(gains[index], integrals)
        poles = torch.abs(poles)
        poles *= -1

        # make sure none of the poles cross the integrator limit
        poles = torch.minimum(poles, torch.full(matrix_size, INTEGRATOR_LIMIT, dtype=self.dtype, device=self.device))

        # convert the poles to gammas
        gammas[index] = torch.exp(poles)
        gammas[index] = torch.where(gammas[index].le(0.00001), 0.00001, gammas[index])


    def forward(self, phi: torch.Tensor):
        if phi.dim() != 1:
            raise ValueError("Input tensor to forward must be 1 dimensional.")
        if phi.size(dim=0) != self.layer_size:
            raise ValueError("Input tensor to forward is the wrong size.")
        if phi.device != self.device:
            raise ValueError("Input tensor to forward does not match device with Ordinet.")
        if phi.dtype != self.dtype:
            raise ValueError("Input tensor to forwrd does not math dtype with Ordinet.")

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
            raise ValueError("x_sequence and loss_gradients sizes do not match on temporal dimension")
        if x_sequence.size(dim=1) != self.layer_size:
            raise ValueError("x_sequence has wrong input size.")
        if loss_gradients.size(dim=1) != self.layer_size:
            raise ValueError("loss_gradients has wrong output size.")
        if x_sequence.device != self.device:
            raise ValueError("x_sequence tensor does not match device with Ordinet.")
        if loss_gradients.device != self.device:
            raise ValueError("loss_gradients tensor does not match device with Ordinet.")
        if x_sequence.dtype != self.dtype:
            raise ValueError("x_sequence tensor does not match device with Ordinet.")
        if loss_gradients.dtype != self.dtype:
            raise ValueError("loss_gradients tensor does not match device with Ordinet.")

        # the length of the sequence we are training on
        sequence_length = x_sequence.size(dim=0)

        # calculate the poles for each gamma
        poles = torch.log(self.gammas)


        # initialize the leading net activations
        leading_activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)
        # initialize gradient-calculating per-transfer values
        trailing_activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

        # the offset gammas to calculate the pole gradients
        offset_gammas = torch.exp(torch.mul(poles, d_offset))
        # the coefficients that normalize the offsets
        offset_coefs = torch.sub(torch.mul(poles, d_offset), poles)
        # offset per-transfer values for calculating pole gradients
        offset_activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)


        # find the maximum forward-search depth
        max_depth = math.floor((-1/INTEGRATOR_LIMIT) * max_tau)

        # find how long it takes for every transfer to decay
        decay_times = torch.floor(max_tau*torch.div(torch.full(self.container_size, 1.0, dtype=self.dtype, device=self.device), -1*poles))
        decay_times = decay_times.type(dtype=torch.int32)
        # find the maximum decay level at the given transfer search depth
        max_decays = torch.exp(torch.mul(poles, decay_times))

        # find the forward-search depth of each transfer ()
        search_indexes = decay_times.clone()
        # search indexes must have the correct row offset for reshaped indexing
        for l in range(search_indexes.size(dim=0)):
            search_indexes[l] *= search_indexes[l].size(dim=0)
            temp_search = search_indexes[l].reshape([search_indexes[l].nelement()])
            temp_search += torch.concat([torch.full([search_indexes[l].size(dim=1)], i, device=self.device) for i in range(search_indexes[l].size(dim=0))])


        # keep track of the backwards loss dependencies
        loss_convolutions = torch.zeros([self.container_size[0], 2, self.container_size[1], self.container_size[2]], dtype=self.dtype, device=self.device)
        # keep track of which loss convolution is on top
        loss_convolution_coeffs = torch.stack([
            torch.zeros(self.container_size, dtype=torch.int32, device=self.device),
            torch.full(self.container_size, 1.0, dtype=torch.int32, device=self.device)
        ], dim=1)
        # keep track of when to switch the loss convolution index
        loss_convolution_tracker = decay_times.clone()


        # the distance between the trailing layer and leading layer at the index
        tail_lengths = [k * max_depth for k in range(self.container_size[0], 0, -1)]

        # contain the layer-wise inputs generated by the leading activations (needed for every intermediate layer)
        input_trails = [None for _ in range(self.container_size[0])]
        for l in range(1, self.num_hidden+1):
            input_trails[l] = torch.zeros([tail_lengths[l], self.layer_size], dtype=self.dtype, device=self.device)

        # contain the layer-wise loss gradients along the bottom staircase
        loss_trails  = [None for _ in range(self.container_size[0])]
        for l in range(1, self.num_hidden+1):
            loss_trails[l] = torch.zeros([max_depth, self.layer_size], dtype=self.dtype, device=self.device)


        # create a padded version of the loss derivitives for overflow
        padded_loss = torch.cat([
            torch.zeros([max_depth, loss_gradients.size(dim=1)], dtype=self.dtype, device=self.device),
            loss_gradients,
            torch.zeros([max(tail_lengths) + max_depth, loss_gradients.size(dim=1)], dtype=self.dtype, device=self.device)
        ])

        print(" --- search dist --- ")
        print(torch.floor(max_tau*torch.div(torch.full(self.container_size, 1.0, dtype=self.dtype, device=self.device), -1*poles)[self.container_size[0]-2]))
        print(" --- max decays ---")
        print(max_decays[self.container_size[0]-2])
        print(" --- gammas ---")
        print(self.gammas[self.container_size[0]-2])
        print(" --- gains ---")
        print(self.gains[self.container_size[0]-2])

        self_grad_time = 0.0
        convolve_grad_time = 0.0
        leader_time = 0.0
        timer = Timer()

        """ Iterate forward through the data sequence """
        for t in range(sequence_length + max(tail_lengths)):

            # backpropogate the loss
            for l in range(self.container_size[0]):

                # get the curr gradient loss here
                curr_loss = None
                # if we are in the first transfer, then we read directly from the loss gradients, otherwise we get it from the loss trail
                if l == self.container_size[0]-1:
                    curr_loss = indp(loss_gradients, t - max_depth)
                else:
                    curr_loss = indw(loss_trails[l+1], t)      

                # get the input to the transfer here
                my_input = None
                # if we are in the first transfer, then we read directly from the input sequence, otherwise we get it from the input trail
                if l == 0:
                    my_input = indw(x_sequence, t)
                else:
                    my_input = indw(input_trails[l], t)

                """ First we take care of the transfer function's current gradient """

                if t - tail_lengths[l] >= 0 and t - tail_lengths[l] < sequence_length:

                    # only if we are within the working range

                    multed_input = torch.mul(self.gains[l], torch.t(my_input))
                    # convolve on the regular activations
                    trailing_activations[l] = torch.mul(trailing_activations[l], self.gammas[l])
                    trailing_activations[l] = torch.add(trailing_activations[l], multed_input)
                    # convolve on the offset activations
                    offset_activations[l] = torch.mul(offset_activations[l], offset_gammas[l])
                    offset_activations[l] = torch.add(offset_activations[l], multed_input)

                    # if l == self.container_size[0]-3:
                    #      print(" --- curr loss:")
                    #      print(curr_loss)
                    #      print(" --- loss convs:")
                    #      print(trailing_activations[l])

                    # update the gradients with chain rule
                    self.gain_grads[l] += torch.mul(torch.divide(trailing_activations[l], self.gains[l]), curr_loss)
                    self.pole_grads[l] += torch.mul(torch.div(torch.sub(offset_activations[l], trailing_activations[l]), offset_coefs[l]), torch.t(curr_loss))

                """ Second we update and save the transfer function's loss convolution """

                # we only do this if it isn't the first transfer, and if we are within range
                if l > 0 and t - tail_lengths[l] + max_depth >= 0 and t - tail_lengths[l] < sequence_length:

                    new_loss = None
                    # if this is the last transfer then treat it a little different
                    if l < self.container_size[0]-1:
                        # take the t modulus into account
                        search_locations = torch.remainder(search_indexes[l] + search_indexes[l].size(dim=0)*t, loss_trails[l+1].size(dim=0))
                        # grab the loss value that is being added to the convolution
                        reshaped_search = search_locations.reshape([search_locations.nelement()]) # turn the search tensor to 1D
                        new_loss = torch.index_select(loss_trails[l+1].reshape(loss_trails[l+1].nelement()), 0, reshaped_search).reshape(search_indexes[l].shape) # get the loss at the index and convert back to shape
                    else:
                        # take the t delta into account
                        search_locations = search_indexes[l] + search_indexes[l].size(dim=0) * t
                        # grab the loss from the loss gradient vector directly (assumes zero padding)
                        reshaped_search = search_locations.reshape([search_locations.nelement()]) # turn the search tensor to 1D
                        new_loss = torch.index_select(padded_loss.reshape(padded_loss.nelement()), 0, reshaped_search).reshape(search_indexes[l].shape) # get the loss at the index and convert back to shape

                    # if l == self.container_size[0]-2:
                    #     print("\n ----- \n")
                    #     print(" --- loss trail:")
                    #     print(loss_trails[l+1])

                    # add this new loss to the loss convolutions
                    multed_new_loss = torch.mul(max_decays[l], new_loss)
                    loss_convolutions[l][0] += torch.mul(loss_convolution_coeffs[l][0], multed_new_loss)
                    loss_convolutions[l][1] += torch.mul(loss_convolution_coeffs[l][1], multed_new_loss)

                    # sum up the loss convolutions into the input, account for the input gradient, then save it
                    saving_loss = torch.mul(loss_convolutions[l][0] + loss_convolutions[l][1], self.gains[l]) # don't forget the gains

                    # if l == self.container_size[0]-1:
                    #     print(" --- loss convs:")
                    #     print(saving_loss)

                    if l == 2:
                        print("before")
                        print(loss_convolutions[l])

                    saving_loss = torch.sum(saving_loss, dim=0)
                    saving_loss = torch.mul(saving_loss, self.act_grad_func(my_input))
                    loss_trails[l][indw_i(loss_trails[l], t)] = saving_loss

                    # remove the backward convolve the loss convolutions to prepare for next
                    loss_convolutions[l][0] -= torch.mul(loss_convolution_coeffs[l][1], curr_loss)
                    loss_convolutions[l][1] -= torch.mul(loss_convolution_coeffs[l][0], curr_loss)

                    loss_convolution_tracker[l] -= 1
                    loss_must_clear = loss_convolution_tracker[l].eq(0)
                    loss_convolution_tracker[l] = torch.where(loss_must_clear, decay_times[l], loss_convolution_tracker[l])
                    loss_convolutions[l][0] = torch.where(torch.logical_and(loss_must_clear, loss_convolution_coeffs[l][0].le(0.5)), 0, loss_convolutions[l][0])
                    loss_convolutions[l][1] = torch.where(torch.logical_and(loss_must_clear, loss_convolution_coeffs[l][1].le(0.5)), 0, loss_convolutions[l][1])

                    temp = loss_convolution_coeffs[l][0].clone()
                    loss_convolution_coeffs[l][0] = torch.where(loss_must_clear, loss_convolution_coeffs[l][1], loss_convolution_coeffs[l][0])
                    loss_convolution_coeffs[l][1] = torch.where(loss_must_clear, temp, loss_convolution_coeffs[l][1])

                    if l == 2:
                        print("subbed")
                        print(loss_convolutions[l])

                    # remove the backward convolve the loss convolutions to prepare for next
                    loss_convolutions[l][0] = torch.div(loss_convolutions[l][0], self.gammas[l])
                    loss_convolutions[l][1] = torch.div(loss_convolutions[l][1], self.gammas[l])

                    if l == 2:
                        print("Final")
                        print(loss_convolutions[l])

                # otherwise we just put zero
                elif l > 0:
                    loss_trails[l][indw_i(loss_trails[l], t)] = torch.zeros(loss_trails[l][indw_i(loss_trails[l], t)].size(), dtype=self.dtype, device=self.device)


            # do the leading forward pass (only if font is within data sequence)
            if t < sequence_length:
                
                # get the input at t
                input_leader = x_sequence[t].clone()

                # do a forward pass
                for b in range(self.container_size[0]):
                    leading_activations[b] = torch.mul(leading_activations[b], self.gammas[b])
                    multed_leader = torch.mul(self.gains[b], torch.t(input_leader))
                    leading_activations[b] = torch.add(leading_activations[b], multed_leader)

                    # we don't care about the final output TODO: take into account last activation gradient for loss
                    if b < self.container_size[0]-1:
                        input_leader = torch.sum(leading_activations[b], dim=1)
                        input_leader = self.act_func(input_leader)

                        # this should always evaluate to a hidden layer, so save the input to the trail
                        input_trails[b+1][indw_i(input_trails[b+1], t)] = input_leader.clone() # overwrites the last input that was read
                        
