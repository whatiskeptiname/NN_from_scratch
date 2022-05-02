import numpy as np
import random
from tools import mnist_loader


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_pos(z):
    return 1.0 / (1.0 + np.exp(z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_pos_prime(z):
    """Derivative of the sigmoid function."""
    return -(sigmoid_pos(z) * (1 - sigmoid_pos(z)))


class Network:
    def __init__(
        self,
        sizes,
        activation_function=sigmoid,
        activation_function_prime=sigmoid_prime,
    ):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # Take the activation function as an argument and initialize it as a class method
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            cost_arr = []  # Keep track of the cost for each epoch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                cost_arr.append(self.cost(mini_batch))
            if test_data:
                test_eval = self.evaluate(test_data)
                print(
                    f"Epoch {j}: {test_eval} / {n_test} Accuracy: {round(100*(test_eval/n_test), 2)} Cost: {np.mean(cost_arr)}"
                )
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(
            activations[-1], y
        ) * self.activation_function_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_function_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost(self, mini_batch):
        """Return the cost of the network for each epoch"""
        square_error = [sum((y - self.feedforward(x)) ** 2) for x, y in mini_batch]
        return np.mean(square_error) / 2

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives (partial C_x) /
        (partial y) for the output activations."""
        return output_activations - y


training_data, _, test_data = mnist_loader.load_data_wrapper()


nn_with_sigmoid = Network(
    [784, 30, 10], activation_function=sigmoid, activation_function_prime=sigmoid_prime
)
print("\n\nNN with sigmoid activation function:")
nn_with_sigmoid.SGD(training_data, 10, 10, 3.0, test_data=test_data)


nn_with_sigmoid_pos = Network(
    [784, 30, 10],
    activation_function=sigmoid_pos,
    activation_function_prime=sigmoid_pos_prime,
)

print("\n\nNN with sigmoid_pos (+z) activation function:")
nn_with_sigmoid_pos.SGD(training_data, 10, 10, 3.0, test_data=test_data)


nn_with_sigmoid = Network(
    [784, 30, 30, 30, 10],
    activation_function=sigmoid,
    activation_function_prime=sigmoid_prime,
)

print("\n\nNN with sigmoid activation function having 3 hidden layers:")
nn_with_sigmoid.SGD(training_data, 10, 10, 3.0, test_data=test_data)
