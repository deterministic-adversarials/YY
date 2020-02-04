import os
import numpy as np
import tensorflow as tf

from models import model_cifar10
from tools import evaluate, train, load_cifar10


img_size = 32
img_chan = 3
n_classes = 10

batch_size = 128
epochs = 50


print('\nPreparing CIFAR-10 data')

(X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_cifar10()

print('\nConstruction graph')

env = model_cifar10()

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, batch_size=batch_size, 
                                        epochs=epochs, name='model_cifar10')

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)

