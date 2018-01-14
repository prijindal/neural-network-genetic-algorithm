"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm

def train_networks(networks, dataset, percentage_dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, percentage_dataset)
        network.print_network()
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset, percentage_dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset, percentage_dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
            
    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.
    dataset = 'multiply'
    percentage_dataset = 0.01

    # Setup logging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG,
        filename='{0}.log.txt'.format(dataset)
    )

    nn_param_choices = {
        'nb_neurons': [2, 4, 8],
        'nb_layers': [1, 2],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
        'loss': [
                    'mean_squared_error', 'mean_squared_logarithmic_error',
                    # 'mean_absolute_error', 'mean_absolute_percentage_error', 
                    # 'squared_hinge', 'hinge', 'categorical_hinge',
                    # 'logcosh',
                    # 'categorical_crossentropy',
                    # 'sparse_categorical_crossentropy', 'binary_crossentropy',
                    # 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'
                ],
        'regularizer': ['l1', 'l2', 'l1_l2'],
        'regularizer_alpha': [0, 0.01, 0.02],
        'initializer': ['zeros', 'ones', 'random_normal', 'random_uniform', 'truncated_normal', 
                        #'variance_scaling', 
                        #'orthogonal', 'identity', 
                        'glorot_normal', 'glorot_uniform',
                        'lecun_uniform', 'lecun_normal',
                        'he_normal', 'he_uniform',]
    }

    # We can add more parameters here
    # 1. Initializers
    # 2. Regularizers
    # 3. Constraints
    # 4. Loss/Objective Function
    # Also we can add parameters for the optimizer function:
    # Learning rate,
    # clipnorm,
    # clipvalue,
    # momentum
    # decay etc 

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset, percentage_dataset)

if __name__ == '__main__':
    main()
