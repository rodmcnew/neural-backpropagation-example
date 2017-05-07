#!/usr/bin/python3
import numpy as np

input_count = 2
hidden_count = 4
output_count = 1
learning_rate = 0.5

input_node_count = input_count + 1  # Add a bias node
output_gradients = np.random.randn(output_count * input_node_count).reshape(input_node_count, output_count)

output_neuron_gradients = np.zeros(output_count)
output_weight_updates = np.zeros(shape=(input_node_count, output_count))


def get_target_output(inputs):
    result = np.zeros(output_count)
    if inputs[0] == 1:  # If the first input value is 1, return 1, else return 0
        result[0] = 1
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    for training_session in range(10000):
        inputs = np.random.randint(2, size=input_node_count)

        # Forward pass
        inputs[input_count] = 1  # the bias node for the hidden layer

        output_out = np.matmul(inputs, output_gradients)
        for o in range(len(output_out)):
            output_out[o] = sigmoid(output_out[o])

        target_output = get_target_output(inputs)
        error = np.subtract(output_out, target_output)
        print('error', round(error[0], 2), 'output', round(output_out[0], 2), 'target', target_output[0], 'input',
              inputs)

        # Calc output neuron error gradients
        for o in range(output_count):
            output = output_out[o]
            target_output = target_output[o]
            output_neuron_gradients[o] = (output_out - target_output) * output * (1 - output_out)

        # Calc output weight updates
        for o in range(output_count):
            for i in range(input_node_count):
                output_weight_updates[i][o] = -learning_rate * inputs[i] * output_neuron_gradients[o]

        # Update output weights
        for o in range(output_count):
            for i in range(input_node_count):
                output_gradients[i][o] += output_weight_updates[i][o]


main()
