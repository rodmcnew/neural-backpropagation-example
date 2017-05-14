import random

import numpy as np
from deep_network import DeepNetwork

from activation_functions import Sigmoid, LeakyRelu

# Train the network to give us the XOR on neuron 0 and the OR on neuron 1
training_data = [
    [[0, 0], [0, 0]],
    [[0, 1], [1, 1]],
    [[1, 0], [1, 1]],
    [[1, 1], [0, 1]]
]

# network = DeepNetwork(2, 4, 1, LeakyRelu(), 0.03)
network = DeepNetwork(2, 4, 2, Sigmoid(), 0.5)

for training_session in range(20000):
    training_set = random.choice(training_data)
    inputs = training_set[0]
    target_output = training_set[1]
    outputs = network.feed_forward(inputs)
    network.back_propagate(inputs, outputs, target_output)
    error = np.subtract(outputs, target_output)
    print('error:', ['{:.4f}'.format(abs(error[0])), '{:.4f}'.format(abs(error[1]))], 'target_output', target_output,
          'output:', outputs)
