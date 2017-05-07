#!/usr/bin/python3
import numpy as np

input_count = 2
output_count = 1
learning_rate = 0.5

input_node_count = input_count + 1  # Add a bias node
weights = np.random.randn(output_count * input_node_count).reshape(input_node_count, output_count)

neuron_gradients = np.zeros(output_count)
weight_updates = np.zeros(shape=(input_node_count, output_count))


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
        inputs[input_count] = 1  # Add the bias node

        output_out = np.matmul(inputs, weights)
        for o in range(len(output_out)):
            output_out[o] = sigmoid(output_out[o])

        target_output = get_target_output(inputs)
        error = np.subtract(output_out, target_output)
        print('error', round(error[0], 2), 'output', round(output_out[0], 2), 'target', target_output[0])

        # Learn
        for o in range(output_count):

            # Calculate this neuron's error gradient
            output = output_out[o]
            target_output = target_output[o]
            neuron_error_gradient = (output_out - target_output) * output * (1 - output_out)

            for i in range(input_node_count):
                # Update this weight to behave better in the future
                weights[i][o] += -learning_rate * inputs[i] * neuron_error_gradient


main()
