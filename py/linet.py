
from linet_helpers import *

import math
import torch

# maximum number of steps before an impulse must decay
MAX_SETTLING_TIME = 128
# maximum number of time-constants that are treated as full decay
DEFAULT_MAX_TAU = 6
# the maxmimum pole location based on MAX_SETTLING_TIME and DEFAULT_MAX_TAU
INTEGRATOR_LIMIT = -(1/(MAX_SETTLING_TIME/DEFAULT_MAX_TAU))
# The maximum gradient that can be applied to avoid divergence
GRADIENT_CLIP = 1000


class LiNet:

    def __init__(self, layer_size: int, num_hidden: int, device: torch.device=torch.device("cpu"), dtype: torch.dtype=torch.float32, act_func: str='ReLU', last_activates: bool=True):
        """
        Initialize an linet recurrent neural network.
        
        :param layer_size: Number of nodes per layer (this currently includes input and output layers TODO: fix this)
        :param num_hidden: Number of hidden layers
        :param device: torch.device to operate on (default: CPU)
        :param dtype: torch.type to use (default: float32)
        :param act_func: string denoting the activation function to use, see linet_helpers.ACT_FUNCS
        :param last_activates: Whether to apply the activation function to the output layer (default False TODO: This is currently disabled)
        """
        
        # TODO: fix last_activates
        last_activates = False

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

        # keep track of forward activations (mostly for debugging)
        self.forward_hook_ids = set()
        self.forward_hooks = {}

        
    def save(self, gamma_path: str, gain_path: str) -> None:
        """
        Load a model checkpoint from a given path.

        :param gamma_path: Path to the gamma tensor checkpoint
        :param gamma_path: Path to the gain tensor checkpoint
        """
        torch.save(self.gains, gain_path)
        torch.save(self.gammas, gamma_path)
    
    def load(self, gamma_path: str, gain_path: str) -> None:
        """
        Load a model checkpoint from a given path.

        :param gamma_path: Path to the gamma tensor checkpoint
        :param gamma_path: Path to the gain tensor checkpoint
        """
        self.gains = torch.load(gain_path)
        self.gammas = torch.load(gamma_path)
        
        
    def _init_layer(self, gains: torch.Tensor, gammas: torch.Tensor, index: int) -> None:
        """
        Mutate the given layer of a 3D gain and gamma matrix with initialization values.

        :param gains: 3D tensor representing the gain values
        :param gammas: 3D tensor representing the gamma values
        :param index: Which layer of the weights to initializee
        """

        # check dimensions
        if gains[index].size() != gammas[index].size():
            raise ValueError("Layer initialization has gain and gamma tensor sizes that do not match.")
        if gains[index].dim() != 2:
            raise ValueError("Layer initialization should be done on an indexed 2D layer.")
        
        # number of inputs the layer takes
        n_inputs = gains[index].size(dim=1)
        # tuple representing the size of the weight amtrix
        matrix_size = gains[index].size(dim=0), n_inputs

        # He initialization on integral and gain
        integrals = torch.normal(0, math.sqrt(4/n_inputs), matrix_size, dtype=self.dtype, device=self.device)
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


    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Do a forward pass through the network. Uses abnd modifies the current state of the network.

        :param phi: The input vector to the network

        :return: The output vector of the pass
        """

        # check dimensionality
        if phi.dim() != 1:
            raise ValueError("Input tensor to forward must be 1 dimensional.")
        if phi.size(dim=0) != self.layer_size:
            raise ValueError("Input tensor to forward is the wrong size.")
        if phi.device != self.device:
            raise ValueError("Input tensor to forward does not match device with LiNet.")
        if phi.dtype != self.dtype:
            raise ValueError("Input tensor to forwrd does not math dtype with LiNet.")

        # make sure we don't modify this
        y = phi.clone()

        # iterate through every layer
        for b in range(self.container_size[0]):

            # convolve weights
            self.activations[b] = torch.mul(self.activations[b], self.gammas[b])
            self.activations[b] = torch.add(self.activations[b], torch.mul(self.gains[b], torch.t(y)))

            # sum weights into next layer
            y = torch.sum(self.activations[b], dim=1)

            # apply activation function
            if b != self.container_size[0]-1 or self.last_activates:
                y = self.act_func(y)

            # store to hook for debugging
            if b in self.forward_hook_ids:
                self.forward_hooks[b] = y.clone()

        # output of final layer is return value
        return y

    
    def reset(self):
        """
        Reset the state of the recurrent network. This must be dont between independent forward passes.
        """
        self.activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

    
    def create_forward_hook(self, i: int) -> None:
        """
        Add an index to the list of layers that store a forward hook.

        :param i: Index of the layer to create a hook for
        """
        if i not in self.forward_hook_ids:
            # add i to dict
            self.forward_hook_ids.add(i)
            self.forward_hooks[i] = None

    def remove_forward_hook(self, i: int):
        """
        Remove an index to the list of layers that store a forward hook.
        (Does nothing if i is not a hook)

        :param i: Index of the layer remove
        """
        if i in self.forward_hook_ids:
            # remove from dict
            self.forward_hook_ids.remove(i)
            self.forward_hooks.pop(i)

    def get_forward_hook(self, i: int) -> torch.Tensor:
        """
        Get the value of the forward hook at index i, if it is stored. Returns None of the hook has not been set.

        :param i: Index of the layer to get.

        :return: The previous value of that layer.
        """
        if i not in self.forward_hook_ids:
            # check that this is a valid hook
            raise ValueError("Indexed invalid forward hook.")
        return self.forward_hooks[i]


    def backward(self, x_sequence: torch.Tensor, loss_gradients: torch.Tensor, max_tau: int=DEFAULT_MAX_TAU, d_offset: float=0.99, lr=None, momentum=0.0) -> None:
        """
        Perform Backpropogration-Forward-Through-Time on a sequence of inputs and loss gradients.
        Note that loss must be input as the gradient, rather than the actual loss.

        :param x_sequence: 3D tensor with the first axis as time representing the input data.
        :param loss_gradients: 3D tensor with the first axis as tiem representing the loss gradients to minimize.
        :param max_tau: The number of time constants to treat as decay (default MAX_TAU)
        :param d_offset: The offset of the second pole, as a percentage, when calculating pole gradients. (default 0.99)
        :param lr: The learning rate to use for in-sequence stochastic gradient descent (not fully implemented and might cause problems, default None)
        :param momentum: The momentum (simple discounted momentum, defualt 0)
        """

        # check dimensions
        if x_sequence.size(dim=0) != loss_gradients.size(dim=0):
            raise ValueError("x_sequence and loss_gradients sizes do not match on temporal dimension")
        if x_sequence.size(dim=1) != self.layer_size:
            raise ValueError("x_sequence has wrong input size.")
        if loss_gradients.size(dim=1) != self.layer_size:
            raise ValueError("loss_gradients has wrong output size.")
        if x_sequence.device != self.device:
            raise ValueError("x_sequence tensor does not match device with LiNet.")
        if loss_gradients.device != self.device:
            raise ValueError("loss_gradients tensor does not match device with LiNet.")
        if x_sequence.dtype != self.dtype:
            raise ValueError("x_sequence tensor does not match device with LiNet.")
        if loss_gradients.dtype != self.dtype:
            raise ValueError("loss_gradients tensor does not match device with LiNet.")

        # reset to clear any previous passes
        self.reset()

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

                    # update the gradients with chain rule
                    self.gain_grads[l] += torch.mul(torch.divide(trailing_activations[l], self.gains[l]), curr_loss[:, None])
                    self.pole_grads[l] += torch.mul(torch.div(torch.sub(offset_activations[l], trailing_activations[l]), offset_coefs[l]), torch.t(curr_loss))

                    if lr != None:
                        self.gains[l] += lr*self.gain_grads[l]
                        self.gain_grads[l] *= momentum

                """ Second we update and save the transfer function's loss convolution """

                # we only do this if it isn't the first transfer, and if we are within range
                if l > 0 and t - tail_lengths[l] + max_depth >= 0 and t - tail_lengths[l] < sequence_length:

                    new_loss = None
                    # if this is the last transfer then treat it a little different
                    if l < self.container_size[0]-1:
                        # take the t modulus into account
                        search_locations = torch.remainder(search_indexes[l] + search_indexes[l].size(dim=0)*t, loss_trails[l+1].nelement())
                        # grab the loss value that is being added to the convolution
                        reshaped_search = search_locations.reshape([search_locations.nelement()]) # turn the search tensor to 1D
                        new_loss = torch.index_select(loss_trails[l+1].reshape(loss_trails[l+1].nelement()), 0, reshaped_search).reshape(search_indexes[l].shape) # get the loss at the index and convert back to shape
                    else:
                        # take the t delta into account
                        search_locations = search_indexes[l] + search_indexes[l].size(dim=0) * t
                        # grab the loss from the loss gradient vector directly (assumes zero padding)
                        reshaped_search = search_locations.reshape([search_locations.nelement()]) # turn the search tensor to 1D
                        new_loss = torch.index_select(padded_loss.reshape(padded_loss.nelement()), 0, reshaped_search).reshape(search_indexes[l].shape) # get the loss at the index and convert back to shape

                    # add this new loss to the loss convolutions
                    multed_new_loss = torch.mul(max_decays[l], new_loss)
                    loss_convolutions[l][0] += torch.mul(loss_convolution_coeffs[l][0], multed_new_loss)
                    loss_convolutions[l][1] += torch.mul(loss_convolution_coeffs[l][1], multed_new_loss)

                    # sum up the loss convolutions into the input, account for the input gradient, then save it
                    saving_loss = torch.mul(loss_convolutions[l][0] + loss_convolutions[l][1], self.gains[l]) # don't forget the gains

                    saving_loss = torch.sum(saving_loss, dim=0)
                    saving_loss = torch.mul(saving_loss, self.act_grad_func(my_input))
                    loss_trails[l][indw_i(loss_trails[l], t)] = saving_loss

                    # figure out which losses need to be flipped
                    loss_must_clear = loss_convolution_tracker[l].le(0)
                    loss_convolution_tracker[l] -= 1
                    loss_convolution_tracker[l] = torch.where(loss_must_clear, decay_times[l], loss_convolution_tracker[l])
                    
                    # set those to be flipped to zero
                    loss_convolutions[l][0] = torch.where(torch.logical_and(loss_must_clear, loss_convolution_coeffs[l][0].le(0.5)), 0, loss_convolutions[l][0])
                    loss_convolutions[l][1] = torch.where(torch.logical_and(loss_must_clear, loss_convolution_coeffs[l][1].le(0.5)), 0, loss_convolutions[l][1])

                    # flip the losses
                    temp = loss_convolution_coeffs[l][0].clone()
                    loss_convolution_coeffs[l][0] = torch.where(loss_must_clear, loss_convolution_coeffs[l][1], loss_convolution_coeffs[l][0])
                    loss_convolution_coeffs[l][1] = torch.where(loss_must_clear, temp, loss_convolution_coeffs[l][1])

                    # remove the backward convolve the loss convolutions to prepare for next
                    loss_convolutions[l][0] -= torch.mul(loss_convolution_coeffs[l][1], curr_loss[:, None])
                    loss_convolutions[l][1] -= torch.mul(loss_convolution_coeffs[l][0], curr_loss[:, None])

                    # backward convolve the loss convolutions to prepare for next
                    loss_convolutions[l][0] = torch.div(loss_convolutions[l][0], self.gammas[l])
                    loss_convolutions[l][1] = torch.div(loss_convolutions[l][1], self.gammas[l])

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
                        

    def apply_grads(self, lr_k, lr_p=None, momentum=0.0) -> None:
        """
        Apply the gradients stored in the network to its weights.

        :param lr_k: learning rate of teh gains
        :param lr_p: learning rate of the poles (None if same as lr_k, default None)
        :param momentum: Amount of momentum to store for next time
        """

        # configure lr_p
        if lr_p == None:
            lr_p = lr_k

        # apply gradient clipping
        self.gain_grads = torch.where(torch.gt(self.gain_grads, GRADIENT_CLIP), GRADIENT_CLIP, self.gain_grads)
        self.gain_grads = torch.where(torch.lt(self.gain_grads, -GRADIENT_CLIP), -GRADIENT_CLIP, self.gain_grads)
        self.pole_grads = torch.where(torch.gt(self.pole_grads, GRADIENT_CLIP), GRADIENT_CLIP, self.pole_grads)
        self.pole_grads = torch.where(torch.lt(self.pole_grads, -GRADIENT_CLIP), -GRADIENT_CLIP, self.pole_grads)

        # update weights
        self.gains += lr_k*self.gain_grads
        self.gain_grads *= momentum

        # calculate the poles for each gamma and update
        poles = torch.log(self.gammas)
        poles += lr_p*self.pole_grads
        self.gammas = torch.exp(torch.minimum(poles, torch.full_like(poles, INTEGRATOR_LIMIT)))
        self.pole_grads *= momentum


    def debug_backward(self, x_sequence, loss_gradients, layer_num):
        """
        Use a slow simplified version of backpropogation at a specific layer to check that the real version is worling correctly.

        :param x_sequence: 3D tensor with the first axis as time representing the input data.
        :param loss_gradients: 3D tensor with the first axis as tiem representing the loss gradients to minimize.
        """

        # check dimensions
        if x_sequence.size(dim=0) != loss_gradients.size(dim=0):
            raise ValueError("x_sequence and loss_gradients sizes do not match on temporal dimension")
        if x_sequence.size(dim=1) != self.layer_size:
            raise ValueError("x_sequence has wrong input size.")
        if loss_gradients.size(dim=1) != self.layer_size:
            raise ValueError("loss_gradients has wrong output size.")
        if x_sequence.device != self.device:
            raise ValueError("x_sequence tensor does not match device with LiNet.")
        if loss_gradients.device != self.device:
            raise ValueError("loss_gradients tensor does not match device with LiNet.")
        if x_sequence.dtype != self.dtype:
            raise ValueError("x_sequence tensor does not match device with LiNet.")
        if loss_gradients.dtype != self.dtype:
            raise ValueError("loss_gradients tensor does not match device with LiNet.")
        if layer_num < 0 or layer_num >= self.container_size[0]:
            raise ValueError("invalid debug layer.")

        # the length of the sequence we are training on
        sequence_length = x_sequence.size(dim=0)

        curr_loss_seq = [loss_gradients[i] for i in range(loss_gradients.size(dim=0))]

        for l in range(self.container_size[0]-1, layer_num-1, -1):

            # initialize the leading net activations
            leading_activations = torch.zeros(self.container_size, dtype=self.dtype, device=self.device)

            # keep track of the input sequence for the selected layer
            input_sequence = []

            for t in range(sequence_length):
                    
                # get the input at t
                input_leader = x_sequence[t].clone()

                if l == 0:
                    input_sequence.append(input_leader)

                else:

                    # do a forward pass
                    for b in range(self.container_size[0]):
                        leading_activations[b] = torch.mul(leading_activations[b], self.gammas[b])
                        multed_leader = torch.mul(self.gains[b], torch.t(input_leader))
                        leading_activations[b] = torch.add(leading_activations[b], multed_leader)

                        input_leader = torch.sum(leading_activations[b], dim=1)
                        input_leader = self.act_func(input_leader)
                    
                        if b == l - 1:
                            input_sequence.append(input_leader)
                            break
            
            # keep track of the backward loss convolutions
            loss_convolutions = torch.zeros_like(self.gammas[l])

            loss_sequence = []

            for t in range(sequence_length-1, -1, -1):
                loss_convolutions *= self.gammas[l]
                loss_convolutions += torch.mul(curr_loss_seq[t][:, None], self.gains[l])

                loss_step = torch.sum(loss_convolutions, dim=0)
                loss_step *= self.act_grad_func(input_sequence[t])
                loss_sequence.append(loss_step)
            
            if l == layer_num:
                return loss_sequence[::-1]
            else:
                curr_loss_seq = loss_sequence[::-1]

