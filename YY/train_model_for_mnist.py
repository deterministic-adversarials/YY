import os
import numpy as np
import tensorflow as tf

from models import model_mnist
from tools import evaluate, train, load_mnist


img_size = 28
img_chan = 1
n_classes = 10

batch_size = 128
epochs = 10


print('\nPreparing MNIST data')

(X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_mnist()

print('\nConstruction graph')

env = model_mnist()

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, batch_size=batch_size, 
                                        epochs=epochs, name='model_mnist')

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)

