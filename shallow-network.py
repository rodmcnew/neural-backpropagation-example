import numpy as np

input_count = 2
output_count = 1
learning_rate = 0.5

input_node_count = input_count + 1  # Add a bias node
weights = np.random.randn(output_count * input_node_count).reshape(input_node_count, output_count)


def get_target_output(inputs):
    result = np.zeros(output_count)
    if inputs[0] == 1:  # If the first input value is 1, return 1, else return 0
        result[0] = 1
    return result


def feed_forward(inputs):
    # Ensure the additional bias node always outputs "1"
    inputs[input_count] = 1

    # Multiply the inputs by the weights in each neuron
    outputs = np.matmul(inputs, weights)

    # Apply the sigmoid activation function to each neuron's output
    for o in range(len(outputs)):
        outputs[o] = 1 / (1 + np.exp(-outputs[o]))  # Sigmoid function

    return outputs


def back_propagate(inputs, outputs, target_output):
    for neuron in range(output_count):
        output = outputs[neuron]
        target_output = target_output[neuron]
        # Calculate the error gradient using the derivative of the sigmoid function
        error_gradient = (outputs - target_output) * output * (1 - outputs)

        for weight in range(input_node_count):
            # Update this weight to behave better in the future
            weights[weight][neuron] += -learning_rate * inputs[weight] * error_gradient


def main():
    for training_session in range(10000):
        inputs = np.random.randint(2, size=input_node_count)
        target_output = get_target_output(inputs)
        outputs = feed_forward(inputs)
        back_propagate(inputs, outputs, target_output)
        error = np.subtract(outputs, target_output)
        print('error', round(error[0], 2), 'output', round(outputs[0], 2), 'target', target_output[0])


main()
