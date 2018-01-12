"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10(percentage=1):
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    TRAIN_SIZE = len(x_train) * percentage
    TEST_SIZE = len(x_test) * percentage

    x_train = x_train[:TRAIN_SIZE]
    y_train = y_train[:TRAIN_SIZE]
    x_test = x_test[:TEST_SIZE]
    y_test = y_test[:TEST_SIZE]

    x_train = x_train.reshape(TRAIN_SIZE, 3072)
    x_test = x_test.reshape(TEST_SIZE, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist(percentage=1):
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    TRAIN_SIZE = int(len(x_train) * percentage)
    TEST_SIZE = int(len(x_test) * percentage)

    x_train = x_train[:TRAIN_SIZE]
    y_train = y_train[:TRAIN_SIZE]
    x_test = x_test[:TEST_SIZE]
    y_test = y_test[:TEST_SIZE]

    x_train = x_train.reshape(TRAIN_SIZE, 784)
    x_test = x_test.reshape(TEST_SIZE, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    loss = network['loss']
    regularizer = network['regularizer']
    regularizer_alpha = network['regularizer_alpha']
    initializer = network['initializer']

    model = Sequential()
    
    if regularizer == 'l1':
        regularizer_function = regularizers.l1
    elif regularizer == 'l2':
        regularizer_function = regularizers.l2
    elif regularizer == 'l1_l2':
        regularizer_function = regularizers.l1_l2
    else:
        regularizer_function = regularizers.l1

    kernel_regularizer = regularizer_function(regularizer_alpha)

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, kernel_initializer=initializer ,kernel_regularizer=kernel_regularizer, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset, percentage_dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10(percentage_dataset)
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist(percentage_dataset)

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
