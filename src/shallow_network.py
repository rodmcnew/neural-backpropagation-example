import numpy as np


class ShallowNetwork:
    def __init__(self, input_count, output_count, activation_function, learning_rate):
        self.activation_function = activation_function
        self.input_count = input_count
        self.output_count = output_count
        self.learning_rate = learning_rate
        self.input_node_count = input_count + 1  # Add a bias node

        # This will hold the weights for our neurons
        self.weights = np.random.randn(output_count * self.input_node_count).reshape(self.input_node_count,
                                                                                     output_count)

    # Run the network forward to convert inputs to outputs
    def feed_forward(self, inputs):
        # Add the additional bias node that always outputs "1"
        inputs = np.append(inputs, [1])

        # Multiply the inputs by the weights in each neuron
        outputs = np.matmul(inputs, self.weights)

        # Apply the activation function to each neuron's output
        for o in range(len(outputs)):
            outputs[o] = self.activation_function.get_y(outputs[o])

        return outputs

    # Run the network backward to make it learn to produce better output in the future
    def back_propagate(self, inputs, outputs, target_output):
        # Add the additional bias node that always outputs "1"
        inputs = np.append(inputs, [1])

        for neuron_i in range(self.output_count):

            output = outputs[neuron_i]
            target_output = target_output[neuron_i]

            # Calculate the error gradient using the derivative of the activation function
            error_gradient = (output - target_output) * self.activation_function.get_slope(output)

            for input_i in range(self.input_node_count):
                # Update this weight so that it descends down the error gradient
                self.weights[input_i][neuron_i] -= self.learning_rate * inputs[input_i] * error_gradient
