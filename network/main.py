from typing import List, Union

import numpy as np
import math
from modules.network.utils import (sigmoid, sigmoid_der,
                                   linear, linear_der,
                                   mse_loss, mse_loss_der,
                                   softmax, cross_entropy_loss,
                                   cross_entropy_loss_der)


class Elman_Jordan:
    def __init__(self, lr: float,
                 momentum: float,
                 shape: List[int], make_zero_context: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.make_zero_context = make_zero_context

        self.shape = shape

        self.n_layers = len(shape)

        self.layers = self.__init_layers__()
        self.weights = self.__init_weights__()

        self.dw = [0] * len(self.weights)

    def __init_layers__(self) -> List[np.ndarray]:
        layers = list()

        layers.append(np.ones(self.shape[0] + self.shape[-1] + 1))

        for i in range(1, self.n_layers):
            layers.append(np.ones(self.shape[i]))

        return layers

    def __init_weights__(self) -> List[np.ndarray]:
        if len(self.layers) == 0:
            raise ValueError('Before weight initialization, initialize layers!')

        weights = list()

        for i in range(self.n_layers - 1):
            curr_weights = np.random.randn(self.layers[i].size,
                                           self.layers[i + 1].size) * np.sqrt(2 / self.layers[i].size)

            weights.append(curr_weights)

        return weights

    def linear(x: np.ndarray) -> np.ndarray:
        x.astype(int)
        return math.asinh(x)

    def propagate_forward(self, x: Union[np.ndarray, List]) -> np.ndarray:
        self.layers[0][0: self.shape[0]] = x

        if self.make_zero_context:
            self.layers[0][self.shape[0]: -1] = np.zeros_like(self.layers[-1])
        else:
            self.layers[0][self.shape[0]: -1] = self.layers[-1]

        for i in range(1, len(self.shape) - 1):
            self.layers[i][...] = linear(
                np.dot(self.layers[i - 1], self.weights[i - 1])
            )

        if len(self.shape) - 2 >= 0:
            last_idx = len(self.shape) - 1

            self.layers[last_idx][...] = linear(
                np.dot(self.layers[last_idx - 1], self.weights[last_idx - 1])
            )

        return self.layers[-1]

    def propagate_backward(self, target) -> float:
        deltas = list()

        loss_number = mse_loss(y_pred=self.layers[-1],
                               y_true=target)
        last_layer_delta = mse_loss_der(y_pred=self.layers[-1],
                                        y_true=target)

        deltas.append(last_layer_delta)

        for i in range(len(self.shape) - 2, 0, -1):
            curr_delta = np.dot(deltas[0],
                                self.weights[i].T * linear_der(self.layers[i]))

            deltas.insert(0, curr_delta)

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            curr_delta = np.atleast_2d(deltas[i])

            curr_dw = np.dot(layer.T, curr_delta)

            self.weights[i] += self.lr * curr_dw + self.lr * self.momentum * self.dw[i]

            self.dw[i] = curr_dw

        return loss_number
    def train_model(network: Elman_Jordan, dataset: BaseDataset, n_epochs: int) -> List[float]:
        """
        Performs model training

        :param network: network to train
        :param dataset: dataset for training
        :param n_epochs: number of  epochs we want to train
        :return: list of average epoch errors
        """

        tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...')

        total_error_list = list()

        for _ in tqdm_epochs:
            errors_epoch_list = list()

            for input_values, true_prediction in dataset:
                result = network.propagate_forward(x=input_values)

                error = network.propagate_backward(target=true_prediction)

                errors_epoch_list.append(error)

            average_error = sum(errors_epoch_list) / len(errors_epoch_list)

            tqdm_epochs.set_postfix(
                text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
            )

            total_error_list.append(average_error)

        return total_error_list


    def train_model_min_error(network: Elman_Jordan, dataset: BaseDataset,
                            n_epochs: int, min_error: float,
                            verbose: bool = True) -> List[float]:
        """
        Performs model training

        :param network: network to train
        :param dataset: dataset for training
        :param n_epochs: number of  epochs we want to train
        :param min_error: error we want to reach
        :param verbose: if True shows progress bar
        :return: list of average epoch errors
        """

        tqdm_epochs = tqdm(range(n_epochs), postfix=f'Epochs...', disable=not verbose)

        total_error_list = list()

        for _ in tqdm_epochs:
            errors_epoch_list = list()

            for input_values, true_prediction in dataset:
                result = network.propagate_forward(x=input_values)

                error = network.propagate_backward(target=true_prediction)

                errors_epoch_list.append(error)

            average_error = sum(errors_epoch_list) / len(errors_epoch_list)

            if average_error <= min_error:
                break

            tqdm_epochs.set_postfix(
                text=f'Epochs... Average error: {sum(errors_epoch_list) / len(errors_epoch_list):.2f}'
            )

            total_error_list.append(average_error)

        return total_error_list
    
    def eval_model(network: Elman_Jordan, dataset: BaseDataset):
        results_array = list()

        for input_values, true_prediction in dataset:
            result = network.propagate_forward(x=input_values)
            result = result.tolist()

            result = [int(round(curr_result)) for curr_result in result]

            try:
                print(f'Input: {input_values}, Pred num: {result}. True: {true_prediction}')

                if true_prediction == result:
                    results_array.append(1)
                else:
                    results_array.append(0)
            except ValueError as err:
                results_array.append(0)

        resulting_accuracy = sum(results_array) / len(results_array)

        return resulting_accuracy




if __name__ == '__main__':
    network = Elman_Jordan(lr=0.01, momentum=0.1, shape=[10, 15, 15, 1])

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = network.propagate_forward(x=x)
    error = network.propagate_backward(target=1, )

    print(f'Result: {result}')
    print(f'Error: {error}')


