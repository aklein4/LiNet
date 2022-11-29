
import linet
import linet_helpers

import torch
import argparse
import matplotlib.pyplot as plt
import random
import sys


CHECKPOINT_FOLDER = "./model_checkpoints/"


class DataLoader:

    def __init__(self, data: list, labels: list):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        if len(self.data) != len(self.labels):
            raise RuntimeError("Size of data does not match size of labels")
        return len(self.data)
    
    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise RuntimeError("Invalid DataLoader index.")
        return self.data[index], self.labels[index]


def train(net: linet.LiNet, num_epochs: int=-1, training_data: DataLoader=None, validation_data: DataLoader=None,
        learning_rate: float=None, batch_size: int=None, verbose: bool=True, plot: bool=False,
        checkpoint_freq: int=None, classifier: bool=False, shuffle: bool=False) -> None:

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

    train_loss = []
    val_loss = []
    
    # train for basically forever
    epoch = 0
    while epoch < num_epochs or num_epochs == -1:
        epoch += 1

        msg = ""
        if verbose:
            print("\n --- Epoch", epoch, "---")
            sys.stdout.write("Training... ")

        training_inds = [i for i in range(len(training_data))]
        training_inds.reverse()
        if shuffle:
            random.shuffle(training_inds)

        train_loss.append(0)
        done = 0
        while len(training_inds) > 0:
            batch_x = []
            batch_y = []

            while len(batch_x) < batch_size and len(training_inds) > 0:
                ind = training_inds.pop()
                x, y = training_data[ind]
                batch_x.append(x)
                batch_y.append(y)

            for i in range(len(batch_x)):
                net.reset()

                out = None
                for t in range(batch_x[i].shape[0]):
                    out = net.forward(batch_x[i][t])[0]

                pred = last_layer(out)
                train_loss[-1] += (pred - batch_y[i])**2 / len(training_data)

                last_grad = last_layer_diff(out)
                loss_grads = torch.zeros_like(batch_x[i])
                loss_grads[-1][0] = last_grad*(batch_y[i] - pred)
                net.backward(batch_x[i], loss_grads)

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
            
            net.apply_grads(learning_rate / len(batch_x))

        if validation_data is not None:
            val_loss.append(0)
            for i in range(len(validation_data)):
                net.reset()

                x, y = validation_data[i]

                out = None
                for t in range(x.shape[0]):
                    out = net.forward(x[t])[0]
                
                pred = last_layer(out)
                val_loss[-1] += (pred - y)**2 / len(validation_data)

        if verbose:
            sys.stdout.write("\n")
            print("Train Loss:", round(train_loss[-1].item(), 5))
            print("Val Loss:", round(val_loss[-1].item(), 5))
        

        if checkpoint_freq is not None and epoch % checkpoint_freq == 0:
            net.save(CHECKPOINT_FOLDER+"epoch_"+str(epoch)+"_gammas.pt", CHECKPOINT_FOLDER+"epoch_"+str(epoch)+"_gains.pt")

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