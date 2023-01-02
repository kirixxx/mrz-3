"""Module with testing in different datasets"""

from typing import List, Tuple

from modules.data.datasets import (FibonacciDataset,
                                   FactorialDataset,
                                   PeriodDataset,
                                   ExponentialDataset)
from modules.network.jordan_network import Jordan
from modules.help.plots import draw_error_plot, draw_errors_for_all

from modules.train import train_model
from modules.evaluate import eval_model
from config import Config


def train_eval_fibonacci(verbose: bool = False) -> Tuple[float, List[float]]:
    dataset = FibonacciDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                               number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on fibonacci sequence errors')

    return accuracy, errors_list


def train_eval_period(verbose: bool = False) -> Tuple[float, List[float]]:
    dataset = PeriodDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                            number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on periodical sequence errors')

    return accuracy, errors_list


def train_eval_factorial(verbose: bool = False) -> Tuple[float, List[float]]:
    dataset = FactorialDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                               number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on factorial sequence errors')

    return accuracy, errors_list


def train_eval_exponent(verbose: bool = False) -> Tuple[float, List[float]]:
    dataset = ExponentialDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                                 number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan model on exponential sequence errors')

    return accuracy, errors_list


if __name__ == '__main__':
    verbose = True

    accuracy_fibonacci, errors_list_fibonacci = train_eval_fibonacci(verbose=verbose)
    print(f'Fibonacci accuracy: {accuracy_fibonacci}')

    accuracy_period, errors_list_period = train_eval_period(verbose=verbose)
    print(f'Period accuracy: {accuracy_period}')

    accuracy_factorial, errors_list_factorial = train_eval_factorial(verbose=verbose)
    print(f'Factorial accuracy: {accuracy_factorial}')

    accuracy_exponent, errors_list_exponent = train_eval_exponent(verbose=verbose)
    print(f'Exponent accuracy: {accuracy_exponent}')