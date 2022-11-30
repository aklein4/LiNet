
import linet
import linet_helpers

import torch
import argparse
import matplotlib.pyplot as plt
import random
import sys


"""
Training function and data structures for LiNet.
"""


# folder where checkpoints should be stored
CHECKPOINT_FOLDER = "./model_checkpoints/"


class DataLoader:
    """
    Rip off of PyTorch DataLoader (should probably get rid of this and use that instead) that the train 
    function uses to access data. Stores 2 things:
     - a list of the inputs in this data set
     - a list of the corresponding targets (labels or regression values) for this data set
    """

    def __init__(self, data: list, labels: list):
        """
        Initialize the dataloader.
        :param data: List of datapoints
        :ara labels: list of targets such that labels[i] is target of data[i]
        """
        self.data = data # list of time-series data
        self.labels = labels # list of targets
    
    def __len__(self) -> int:
        """
        Get the length of the data set.
        """
        if len(self.data) != len(self.labels):
            raise RuntimeError("Size of data does not match size of labels")
        return len(self.data)
    
    def __getitem__(self, index: int) -> tuple:
        """
        Get the dataset item at a given index, throws error if out of bounds.
        :param index: Index of the item to retrieve.
        """
        if index < 0 or index >= len(self):
            raise RuntimeError("Invalid DataLoader index.")
        return self.data[index], self.labels[index]


def train(net: linet.LiNet, num_epochs: int=-1, training_data: DataLoader=None, validation_data: DataLoader=None,
        learning_rate: float=None, batch_size: int=None, verbose: bool=True, plot: bool=False,
        checkpoint_freq: int=None, classifier: bool=False, shuffle: bool=False) -> None:
    """
    Perform minibatch (or stocastic or batch) gradient descent on the given data using the given LiNet.
    :param net: The network to train
    :param num_epochs: Number of epochs to train for (-1 for infinite, default: -1)
    :param training_data: DataLoader representing the training set
    :param validation_data: DataLoader representing the validation set to check every epoch (optional)
    :param learning_rate: Learning rate to use for training TODO: Seperate lr for gains and poles
    :param batch_size: Size of batch to use for gradient descent (default is full training set)
    :param checkpoint_freq: How often to save a model checkpoint (optional)
    :param classifier: Whether to treat the network as a classifier (applies sigmoid activation to last layer, default: False)
    :param shuffle: Whether to shuffle the training data each epoch (default: False)
    :param verbose: Whether to print progress messages while training (default: True)
    :param plot: Whether to save the loss to a graph every epoch (default: False)
    """

    # check that parameters have been given
    if training_data is None:
        raise RuntimeError("training_data must be provided for training.")
    if learning_rate is None:
        raise RuntimeError("learning_rate must be provided for training.")
    if checkpoint_freq is None:
        RuntimeWarning("No checkpoint frequency provided, model data will not be saved.")
    if num_epochs == -1:
        RuntimeWarning("num_epochs not provided, defaulting to infinite training.")
    
    # set default values
    if batch_size is None:
        batch_size = len(training_data)

    # if this is a classifier then we should give it an extra sigmoid layer
    last_layer = linet_helpers.ACT_FUNCS["identity"]
    last_layer_diff = linet_helpers.DIFF_FUNCS["identity"]
    if classifier:
        last_layer = linet_helpers.ACT_FUNCS["sigmoid"]
        last_layer_diff = linet_helpers.DIFF_FUNCS["sigmoid"]

    # keep track of the losses for plotting
    train_loss = []
    val_loss = []
    
    # train for epochs or forever
    epoch = 0
    while epoch < num_epochs or num_epochs == -1:
        epoch += 1

        # verbose header
        msg = ""
        if verbose:
            print("\n --- Epoch", epoch, "---")
            sys.stdout.write("Training... ")

        # get list of indices we will train on
        training_inds = [i for i in range(len(training_data))]
        training_inds.reverse()
        if shuffle:
            random.shuffle(training_inds)

        # track info for verbose/plot
        train_loss.append(0)
        done = 0

        # train over all data
        while len(training_inds) > 0:
            # list of batch data
            batch_x = []
            batch_y = []

            # grab a batch-sized chunk of data from the training set
            while len(batch_x) < batch_size and len(training_inds) > 0:
                ind = training_inds.pop()
                x, y = training_data[ind]
                batch_x.append(x)
                batch_y.append(y)

            # train on everything in the batch
            # TODO: parellelize this
            for i in range(len(batch_x)):

                # MUST RESET BETWEEN PASSES
                net.reset()

                # forward pass
                out = None
                for t in range(batch_x[i].shape[0]):
                    out = net.forward(batch_x[i][t])[0]

                # get prediction based on whether this is classifier
                pred = last_layer(out)

                # save loss for plotting/verbose
                train_loss[-1] += (pred - batch_y[i])**2 / len(training_data)

                # we must do the gradient of the last activation here because of classifer workaround
                last_grad = last_layer_diff(out)

                # we use MSE loss and only apply it to the first node on the last time step
                # TODO: generalize this
                loss_grads = torch.zeros_like(batch_x[i])
                loss_grads[-1][0] = last_grad*(batch_y[i] - pred)

                # perform backpropogation on loss
                net.backward(batch_x[i], loss_grads)

                # verbose stuff
                done += 1
                if verbose:
                    eraser = ""
                    for _ in range(len(msg)):
                        eraser += '\b'
                    for _ in range(len(msg)):
                        eraser += ' '
                    for _ in range(len(msg)):
                        eraser += '\b'
                    new_msg = str(done)+"/"+str(len(training_data))
                    sys.stdout.write(eraser+new_msg)
                    sys.stdout.flush()
                    msg = new_msg
            
            # apply the accumulated gradients from the entire batch (normed by batch size)
            net.apply_grads(learning_rate / len(batch_x))

        # check the validation loss
        if validation_data is not None:

            # sum over each data point
            val_loss.append(0)
            for i in range(len(validation_data)):

                # MUST RESET BETWEEN PASSES
                net.reset()

                # get a validation point
                x, y = validation_data[i]

                # forward pass
                out = None
                for t in range(x.shape[0]):
                    out = net.forward(x[t])[0]
                
                # get prediction and loss (loss is averaged)
                pred = last_layer(out)
                val_loss[-1] += (pred - y)**2 / len(validation_data)

        # verbose info
        if verbose:
            sys.stdout.write("\n")
            print("Train Loss:", round(train_loss[-1].item(), 5))
            print("Val Loss:", round(val_loss[-1].item(), 5))
        
        # save checkpoint
        # TODO: checkpoint based on performance
        if checkpoint_freq is not None and epoch % checkpoint_freq == 0:
            net.save(CHECKPOINT_FOLDER+"epoch_"+str(epoch)+"_gammas.pt", CHECKPOINT_FOLDER+"epoch_"+str(epoch)+"_gains.pt")

        # save a plot of the loss so far
        if plot:
            plt.clf()
            plt.plot(range(len(train_loss)), [l.item() for l in train_loss], "Orange", label="train_loss")
            if len(val_loss) > 0:
                plt.plot(range(len(val_loss)), [l.item() for l in val_loss], "Blue", label="val_loss")
            plt.legend()
            plt.ylabel("log MSE loss")
            plt.xlabel("Epoch")
            plt.title("Loss Throughout Training")
            plt.savefig("training_loss.png")