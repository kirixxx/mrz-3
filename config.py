class Config:

    learning_rate = 0.000003
    momentum = 0.1
    num_epochs = 10_000

    min_error = 0.05
    data_type = 'fibonacci'
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    make_zero_context = False

    num_of_precalculated_values = 12
    num_of_input_elements = 2
    num_of_output_elements = 1

    num_of_hidden_neurons = 7


def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        y = np.array(y)
        y.resize((self.sizes[-1], 1))

        activation = np.array(x)
        activation.resize((self.sizes[0], 1))
        activations = [activation]
        zs = []

        for w in self.weights:
            # z = np.dot(w, activation)
            z = np.matmul(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.matmul(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = sp * np.matmul(self.weights[-l + 1].transpose(), delta)

            nabla_w[-l] = np.matmul(delta, activations[-l - 1].transpose())
            nabla_w[-l] = np.matmul(delta, activations[-l].transpose())

        return nabla_w
