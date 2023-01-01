from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from module imprort *
from config import Config


def perform_pipeline_for_plots():
    config = Config()

    data_mapping: Dict = {
        'fibonacci': generate_fibonacci_values(number_of_precalculated_values=config.num_of_precalculated_values),
    }

    dataset = CustomDataset(data=data_mapping[config.data_type],
                            number_of_input_elements=config.num_of_input_elements,
                            number_of_output_elements=config.num_of_output_elements)

    in_features = dataset.number_of_input_elements
    out_features = dataset.number_of_output_elements

    network = Jordan-Elman(lr=config.learning_rate,
                     momentum=config.momentum,
                     make_zero_context=config.make_zero_context,
                     shape=[in_features, config.num_of_hidden_neurons, out_features])

    errors_list = train_model_min_error(network=network,
                                        dataset=dataset,
                                        n_epochs=Config.num_epochs,
                                        min_error=Config.min_error,
                                        verbose=True)

    return errors_list


def learning_rate_epochs_plot():
    learning_rate_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]

    num_of_epochs_list = list()

    for curr_learning_rate in tqdm(learning_rate_list, postfix=f'Training networks'):
        Config.learning_rate = curr_learning_rate

        total_error_list = perform_pipeline_for_plots()

        num_of_epochs_list.append(len(total_error_list))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Learning rate vs number of epochs to achieve {Config.min_error} MSE error')

    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Number of epochs')

    sns.lineplot(x=learning_rate_list, y=num_of_epochs_list, ax=ax)

    plt.show()


def number_of_input_values_epochs_plot():
    num_of_input_values_list = [1, 2, 3, 4]

    if test_data:
        n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {j}: loss = {self.evaluate(test_data)}")
            else:
                print("Epoch {0} complete".format(j))

def sequences_num_of_epochs_plot():
    sequences = []
    for i in range(n):
        index = 0
        sequence = ''
        for j in range(length):
            choices = j[index]
            choice = np.random.randint(0, len(choices))
            token, index = choices[choice]

            sequence += token
        if j[i]:
            print(sequence)
        sequences.append(sequence)
    return sequences


def sequences_num_of_epochs_plot():
    datasets_list = ['fibonacci']

    min_errors_list = list()

    for curr_dataset in tqdm(datasets_list, postfix=f'Training networks'):
        Config.data_type = curr_dataset

        total_error_list = perform_pipeline_for_plots()

        min_errors_list.append(round(min(total_error_list), 2))

    print(min_errors_list)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Sequence vs min error achieved for {Config.num_epochs} epochs')

    ax.set_xlabel('Sequence')
    ax.set_ylabel('Min error')

    barplot_sequences = sns.barplot(x=datasets_list, y=min_errors_list, ax=ax)

    for index, curr_min_error in enumerate(min_errors_list):
        barplot_sequences.text(index, curr_min_error,
                               round(curr_min_error, 2), ha='center')

    plt.show()


if __name__ == '__main__':
    learning_rate_epochs_plot()
