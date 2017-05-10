import random

import numpy as np
from shallow_network import ShallowNetwork

from activation_functions import Sigmoid, LeakyRelu

# Train the network to be a binary "AND" function
training_data = [
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [0]],
    [[1, 1], [1]]
]

# network = ShallowNetwork(2, 1, LeakyRelu(), 0.03)
network = ShallowNetwork(2, 1, Sigmoid(), 0.5)

for training_session in range(10000):
    training_set = random.choice(training_data)
    inputs = training_set[0]
    target_output = training_set[1]
    outputs = network.feed_forward(inputs)
    network.back_propagate(inputs, outputs, target_output)
    error = np.subtract(outputs, target_output)
    print('error:', '{:.4f}'.format(abs(error[0])), 'target_output', target_output, 'output:', outputs)
