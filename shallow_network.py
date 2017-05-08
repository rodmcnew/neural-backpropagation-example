import numpy as np


class ShallowNetwork:
    def __init__(self, input_count, output_count, learning_rate):
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

        # Apply the sigmoid activation function to each neuron's output
        for o in range(len(outputs)):
            outputs[o] = 1 / (1 + np.exp(-outputs[o]))  # Sigmoid function

        return outputs

    # Run the network backward to make it learn to produce better output in the future
    def back_propagate(self, inputs, outputs, target_output):
        # Add the additional bias node that always outputs "1"
        inputs = np.append(inputs, [1])

        for neuron in range(self.output_count):

            output = outputs[neuron]
            target_output = target_output[neuron]

            # Calculate the error gradient using the derivative of the sigmoid function
            error_gradient = (outputs - target_output) * output * (1 - outputs)

            for weight in range(self.input_node_count):
                # Update this weight so that it descends down the error gradient
                self.weights[weight][neuron] -= self.learning_rate * inputs[weight] * error_gradient
