import numpy as np
import tensorflow as tf


##
def load_mnist():
    
    print('\nLoading MNIST')

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train, [-1, 28, 28, 1])
    X_train = X_train.astype(np.float32) / 255
    X_test = np.reshape(X_test, [-1, 28, 28, 1])
    X_test = X_test.astype(np.float32) / 255

    to_categorical = tf.keras.utils.to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('\nSpliting data')

    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)


##
def load_cifar10():
    
    print('\nLoading CIFAR10')

    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_train = X_train.astype(np.float32) / 255
    X_test = np.reshape(X_test, [-1, 32, 32, 3])
    X_test = X_test.astype(np.float32) / 255

    to_categorical = tf.keras.utils.to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('\nSpliting data')

    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)

