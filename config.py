class Config:

    learning_rate = 0.000003
    momentum = 0.1
    num_epochs = 10_000

    min_error = 0.05
    data_type = 'fibonacci'
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    make_zero_context = False

    num_of_precalculated_values = 1
    num_of_input_elements = 2
    num_of_output_elements = 1

    num_of_hidden_neurons = 7


def backprop(self, x, y):

    deltas = []
    
    error = target - self.layers[-1]
    delta = error*dsigmoid(self.layers[-1])
    deltas.append(delta)
    
    for i in range(len(self.shape)-2, 0, -1):
        delta = (np.dot(deltas[0], self.weights[i].T) *
                 dsigmoid(self.layers[i]))
        deltas.insert(0, delta)
    
    for i in range(len(self.weights)):
        layer = np.atleast_2d(self.layers[i])
        delta = np.atleast_2d(deltas[i])
        dw = np.dot(layer.T, delta)
        self.weights[i] += lrate*dw + momentum*self.dw[i]
        self.dw[i] = dw
    
    return (error**2).sum()

def generate_fibonacci_values(number_of_precalculated_values: int) -> List[int]:
    data = list()

    for i in range(number_of_precalculated_values + 1):
        if i == 0 or i == 1:
            data.append(1)
        else:
            data.append(data[i - 1] + data[i - 2])

    return data
