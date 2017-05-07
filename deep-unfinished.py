#!/usr/bin/python3
import numpy as np

import time
import math

input_count = 2
hidden_count = 4
output_count = 1
learning_rate = 0.5

input_node_count = input_count + 1  # Add a bias node
# hidden_node_count = hidden_count + 1  # Add a bias node
#
# hidden_weights = np.random.randn(hidden_node_count * input_node_count).reshape(input_node_count, hidden_node_count)
# hidden_weight_errors = np.zeros(shape=(input_node_count, hidden_node_count))
# hidden_errors = np.zeros(hidden_node_count)

# output_gradients = np.random.randn(output_count * hidden_node_count).reshape(hidden_node_count, output_count)
output_gradients = np.random.randn(output_count * input_node_count).reshape(input_node_count, output_count)

output_neuron_gradients = np.zeros(output_count)  # , dtype='float64'
output_weight_updates = np.zeros(shape=(input_node_count, output_count))


def get_target_output(inputs):
    result = np.zeros(output_count)
    # if inputs[0] == 1 and inputs[1] == 0 or inputs[0] == 0 and inputs[1] == 1:  # XOR
    if inputs[0] == 1:  # just 1
        result[0] = 1
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))
#     # return x * (1 - x)


def main():
    for training_session in range(10000):
        inputs = np.random.randint(2, size=input_node_count)

        # print('weights:', hidden_weights)
        # print('biases', hidden_biases)

        # Forward pass
        inputs[input_count] = 1  # the bias node for the hidden layer
        # hidden_out = np.matmul(inputs, hidden_weights)
        # hidden_out[hidden_count] = 1  # the bias node for the output layer
        # for h in range(len(hidden_out)):
        #     hidden_out[h] = sigmoid(hidden_out[h])

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
            # print('calculate_pd_error_wrt_output', pd_error_wrt_output)
            # print('calculate_pd_total_net_input_wrt_input', pd_total_net_input_wrt_input)
            # print('output_neuron_gradients[o]', output_neuron_gradients[o])

        # Calc output weight updates
        for o in range(output_count):
            for i in range(input_node_count):
                output_weight_updates[i][o] = -learning_rate * inputs[i] * output_neuron_gradients[o]
                ####################################

        # Update output weights
        for o in range(output_count):
            for i in range(input_node_count):
                # print('adj1:', output_weight_updates[h][o] * learning_rate)
                if output_weight_updates[i][o] > 1 or output_weight_updates[i][o] < -1:
                    raise Exception('Seems odd')

                output_gradients[i][o] += output_weight_updates[i][o]

                # print('weight', output_gradients[i][o], 'update', output_weight_updates[i][o])

                if not np.isfinite(output_gradients[i][o]):
                    raise Exception('Weight was adjusted to a non number')


                    # TODO convert any counts below to include biases
                    # # Calc hidden neuron errors
                    # for h in range(hidden_count):
                    #     d_error_wrt_hidden_neuron_output = 0
                    #     for o in range(output_count):
                    #         d_error_wrt_hidden_neuron_output += output_neuron_gradients[o] * output_gradients[h][o]
                    #
                    #     hidden_errors[h] = d_error_wrt_hidden_neuron_output + sigmoid_derivative(hidden_out[h])
                    #
                    # # Calc hidden weight errors
                    # for h in range(hidden_count):
                    #     for i in range(input_count):
                    #         hidden_weight_errors[i][h] = hidden_errors[h] * inputs[i]
                    #
                    # ####################################
                    #
                    # # Calc hidden neuron errors
                    # for h in range(hidden_count):
                    #     d_error_wrt_hidden_neuron_output = 0
                    #     for o in range(output_count):
                    #         d_error_wrt_hidden_neuron_output += output_neuron_gradients[o] * output_gradients[h][o]
                    #
                    #     hidden_errors[h] = d_error_wrt_hidden_neuron_output + sigmoid_derivative(hidden_out[h])
                    #
                    # # Calc hidden weight errors
                    # for h in range(hidden_count):
                    #     for i in range(input_count):
                    #         hidden_weight_errors[i][h] = hidden_errors[h] * inputs[i]
                    #
                    # ####################################
                    #
                    # # Update output weights
                    # for o in range(output_count):
                    #     for h in range(hidden_count):
                    #         # print('adj1:', output_weight_updates[h][o] * learning_rate)
                    #         if output_weight_updates[h][o] > 1 or output_weight_updates[h][o] < -1:
                    #             raise Exception('Seems odd')
                    #
                    #         output_gradients[h][o] -= output_weight_updates[h][o] * learning_rate
                    #
                    #         if not np.isfinite(output_gradients[h][o]):
                    #             raise Exception('Weight was adjusted to a non number')

                    # # Update hidden weights
                    # for h in range(hidden_count):
                    #     for i in range(input_count):
                    #         if hidden_weight_errors[i][h] > 1 or hidden_weight_errors[i][h] < -1:
                    #             raise Exception('Seems odd')
                    #
                    #         # print('adj2:', hidden_weight_errors[i][h] * learning_rate)
                    #         hidden_weights[i][h] -= hidden_weight_errors[i][h] * learning_rate
                    #
                    #         if not np.isfinite(hidden_weights[i][h]):
                    #             raise Exception('Weight was adjusted to a non number')
                    #
                    #             # print('output_neuron_gradients:', output_neuron_gradients)
                    #             # print('hidden_errors:', hidden_errors)
                    #             # print('output_weight_updates', output_weight_updates)


main()
