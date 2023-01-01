from typing import Dict, List

import seaborn as sns
import matplotlib.pyplot as plt
from module imprort *
from config import Config


def plot_errors(errors_list: List[float], sequence: str):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Learning rate vs number of epochs to achieve {Config.min_error} MSE error. Sequence: {sequence}')
    
    error = sequence - errors_list[-1]
    delta = error * ax.sigmoid(layers[-1])
    fig.append(delta)

    plt.show()


if __name__ == '__main__':
    config = Config()

    data_mapping: Dict = {
        'fibonacci': generate_fibonacci_values(number_of_precalculated_values=config.num_of_precalculated_values),
    }

    dataset = CustomDataset(data=data_mapping[config.data_type],
                            number_of_input_elements=config.num_of_input_elements,
                            number_of_output_elements=config.num_of_output_elements)

    in_features = dataset.number_of_input_elements
    out_features = dataset.number_of_output_elements

    network = Jordan(lr=config.learning_rate,
                     momentum=config.momentum,
                     make_zero_context=config.make_zero_context,
                     shape=[in_features, config.num_of_hidden_neurons, out_features])

    errors_list = train_model_min_error(network=network,
                                        dataset=dataset,
                                        n_epochs=config.num_epochs,
                                        min_error=config.min_error)

    accuracy = eval_model(network=network,
                          dataset=dataset)

    plot_errors(errors_list=errors_list, sequence=config.data_type)

    print(f'Accuracy: {accuracy}')
