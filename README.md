# LiNet

LiNet is a neural net architecture that replaces the weights of a standard feed-forward network with linear transfer functions. This allows the network to efficiently operate on online time-series data more efficiently than existing convolutional approaches, and train more efficiently than existing recurrent approaches.

### TODO:
 - Fix activation function on output layer of network (currently worked around in training function)
 - Add option for bias value in transfer function layers.
 - Modularize the network to allow more configurable layer sizes, activation functions, and mix in other types of layers.
 - Use imaginary axis to implement double-pole transfer functions.
 - Add feedback/long-term-memory connections to generalize away from feed-forward convolution.

Currently implemented using single-pole (and evantually double-pole) transfer functions.
