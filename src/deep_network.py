import numpy as np


class DeepNetwork:
    def __init__(self, input_count, hidden_count, output_count, activation_function, learning_rate):
        self.activation_function = activation_function
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.output_count = output_count
        self.learning_rate = learning_rate
        self.input_node_count = input_count + 1  # Add a bias node
        self.hidden_node_count = hidden_count + 1  # Add a bias node

        # Holds the weights of the neurons
        self.hidden_weights = np.random.randn(self.input_node_count * hidden_count)
        self.hidden_weights = self.hidden_weights.reshape(self.input_node_count, hidden_count)
        self.output_weights = np.random.randn(self.hidden_node_count * output_count)
        self.output_weights = self.output_weights.reshape(self.hidden_node_count, output_count)

        # Holds the weight updates during back-propagation
        self.hidden_weight_deltas = np.zeros_like(self.hidden_weights)
        self.output_weight_deltas = np.zeros_like(self.output_weights)

        # Holds the output from the hidden layer which is needed during back-propagation
        self.hidden_outputs = np.zeros(self.hidden_node_count)

    # Run the network forward to convert inputs to outputs
    def feed_forward(self, inputs):
        # Add the additional bias node that always outputs "1"
        inputs = np.append(inputs, [1])

        # Multiply the inputs by the weights in each neuron
        self.hidden_outputs = np.matmul(inputs, self.hidden_weights)

        # Apply the activation function to each neuron's output
        for o in range(len(self.hidden_outputs)):
            self.hidden_outputs[o] = self.activation_function.get_y(self.hidden_outputs[o])

        # Add the additional bias node that always outputs "1"
        self.hidden_outputs = np.append(self.hidden_outputs, [1])

        # Multiply the inputs by the weights in each neuron
        outputs = np.matmul(self.hidden_outputs, self.output_weights)

        # Apply the activation function to each neuron's output
        for o in range(len(outputs)):
            outputs[o] = self.activation_function.get_y(outputs[o])

        return outputs

    # Run the network backward to make it learn to produce better output in the future
    def back_propagate(self, inputs, outputs, target_output):
        # Add the additional bias node that always outputs "1"
        inputs = np.append(inputs, [1])

        output_neuron_error_gradients = np.zeros(self.output_count)

        # Calculate deltas for output neurons
        for neuron_i in range(self.output_count):
            output = outputs[neuron_i]
            target_output = target_output[neuron_i]

            # Calculate the error gradient using the derivative of the activation function
            error_gradient = (output - target_output) * self.activation_function.get_slope(output)

            for input_i in range(self.hidden_node_count):
                # Update this weight so that it descends down the error gradient
                delta = -self.learning_rate * self.hidden_outputs[input_i] * error_gradient
                self.output_weight_deltas[input_i][neuron_i] = delta

            # Save this so we can use it in the hidden layer delta calculation
            output_neuron_error_gradients[neuron_i] = error_gradient

        # Calculate deltas for hidden neurons
        for neuron_i in range(self.hidden_count):
            e_wrt_oj = 0
            for next_neuron_i in range(self.output_count):
                e_wrt_oj += output_neuron_error_gradients[next_neuron_i] * self.output_weights[neuron_i][next_neuron_i]

            output = self.hidden_outputs[neuron_i]

            # Calculate the error gradient using the derivative of the activation function
            error_gradient = e_wrt_oj * self.activation_function.get_slope(output)

            for input_i in range(self.input_node_count):
                # Update this weight so that it descends down the error gradient
                delta = -self.learning_rate * inputs[input_i] * error_gradient
                self.hidden_weight_deltas[input_i][neuron_i] = delta

        # Apply deltas for output neurons
        for neuron_i in range(self.output_count):
            for input_i in range(self.hidden_node_count):
                self.output_weights[input_i][neuron_i] += self.output_weight_deltas[input_i][neuron_i]

        # Apply deltas for hidden neurons
        for neuron_i in range(self.hidden_count):
            for input_i in range(self.input_node_count):
                self.hidden_weights[input_i][neuron_i] += self.hidden_weight_deltas[input_i][neuron_i]
