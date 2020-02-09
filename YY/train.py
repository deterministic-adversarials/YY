import os
import argparse
import numpy as np
import tensorflow as tf

from models import cnn_1, cnn_2, vgg16, vgg19
from tools import evaluate, train
from tools import load_mnist, load_cifar10, load_cifar100


def main(args):

    print('\nPreparing {} data'.format(args.dataset))
    if args.dataset == 'mnist':
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_mnist()
    elif args.dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_cifar10()
    elif args.dataset == 'cifar100':
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_cifar100()

    print('\nConstruction graph')
    if args.model == 'cnn_1':
        env = cnn_1(args)
    elif args.model == 'cnn_2':
        env = cnn_2(args)
    elif args.model == 'vgg16':
        env = vgg16(args)
    elif args.model == 'vgg19':
        env = vgg19(args)

    print('\nInitializing graph')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('\nTraining')
    name = '{0}_{1}'.format(args.model, args.dataset)
    train(sess, env, X_train, y_train, X_valid, y_valid, batch_size=args.batch_size,
                                            epochs=args.epochs, name=name)

    print('\nEvaluating on clean data')
    evaluate(sess, env, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['mnist', 'cifar10', 'cifar100'], default='mnist')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-m', '--model', choices=['cnn_1', 'cnn_2', 'vgg16', 'vgg19'], default='cnn_1')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        args.img_size = 28
        args.img_chan = 1
        args.n_classes = 10
    elif args.dataset == 'cifar10':
        args.img_size = 32
        args.img_chan = 3
        args.n_classes = 10
    elif args.dataset == 'cifar100':
        args.img_size = 32
        args.img_chan = 3
        args.n_classes = 100

    main(args)


