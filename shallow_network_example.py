from shallow_network import ShallowNetwork
import random
import numpy as np

# We will train the network to return "1" only if the first input is "1"
training_data = [
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [1]],
    [[1, 1], [1]]
]

network = ShallowNetwork(2, 1, 0.5)

for training_session in range(20000):
    training_set = random.choice(training_data)
    inputs = training_set[0]
    target_output = training_set[1]
    outputs = network.feed_forward(inputs)
    network.back_propagate(inputs, outputs, target_output)
    error = np.subtract(outputs, target_output)
    print('error:', abs(round(error[0], 2)))
