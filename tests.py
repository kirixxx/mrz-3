from typing import List, Tuple
from module imprort *
from config import Config


def train_eval_fibonacci(verbose: bool = False) -> Tuple[float, List[float]]:
    dataset = FibonacciDataset(number_of_precalculated_values=Config.num_of_precalculated_values,
                               number_of_input_elements=Config.num_of_input_elements)

    in_features = dataset.number_of_input_elements
    out_features = 1

    network = Jordan-Elman(lr=Config.learning_rate,
                     momentum=Config.momentum,
                     shape=[in_features, Config.num_of_hidden_neurons, out_features])

    errors_list = train_model(network=network,
                              dataset=dataset,
                              n_epochs=Config.num_epochs)
    accuracy = eval_model(network=network,
                          dataset=dataset)

    if verbose:
        draw_error_plot(errors_list=errors_list, title='Jordan-Elman model on fibonacci sequence errors')

    return accuracy, errors_list


if __name__ == '__main__':
    verbose = True

    accuracy_fibonacci, errors_list_fibonacci = train_eval_fibonacci(verbose=verbose)
    print(f'Fibonacci accuracy: {accuracy_fibonacci}')