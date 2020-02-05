import os
import argparse
import numpy as np
import tensorflow as tf
import statistics

from attacks import YY
from models import cnn_1, cnn_2, vgg16
from tools import evaluate, predict, pseudorandom_target, exclude_miss
from tools import load_mnist, load_cifar10, load_cifar100
from tools import print_10x10


def make_adv(args, sess, env, X_data, y_data):

    print('\nMaking adversarials')

    X_adv = np.empty((args.n_classes, args.n_classes, args.img_size, args.img_size, args.img_chan))
    X_adv = np.squeeze(X_adv)

    z0 = np.argmax(y_data, axis=1)
    z1 = np.argmax(predict(sess, env, X_data), axis=1)
    ind = z0 == z1

    X_data = X_data[ind]
    labels = z0[ind]
    
    for source in np.arange(args.n_classes):
        print('\n********************Source label {0}********************\n'.format(source))

        X_i = X_data[labels == source]

        for i, xi in enumerate(X_i):
            found = True
            xi = xi[np.newaxis, :] # np.stackと同じ

            for target in np.arange(args.n_classes):
                print(' [{0}/{1}] {2} -> {3}'
                    .format(i+1, X_i.shape[0], source, target), end='')

                if source == target:
                    xadv = xi.copy()
                else:
                    xadv, _, _, _ = YY(args, sess, env, xi, target)

                yadv = predict(sess, env, xadv)
                label = np.argmax(yadv.flatten())
                found = target == label

                if not found:
                    print(' Fail')
                    break

                X_adv[source, target] = np.squeeze(xadv)
                print(' res: {0} {1:.2f}'.format(label, np.max(yadv)))

            if found:
                break      
            
    return X_adv


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

    print('\nInitializing graph')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('\nLoading saved model')
    name = '{0}_{1}'.format(args.model, args.dataset)
    env.saver.restore(sess, 'models/{0}/{1}'.format(name,name))

    print('\nEvaluating on clean data')
    evaluate(sess, env, X_test, y_test)

    #print('\nExcluding misclassification samples')
    #(X_test, y_test) = exclude_miss(sess, env, X_test, y_test, 0, len(X_test))
    #evaluate(sess, env, X_test, y_test)

    print('\nGenerating adversarial data')
    X_adv = make_adv(args, sess, env, X_test, y_test)

    #print('\nEvaluating on adversarial data')
    #evaluate(sess, env, X_adv, y_test)

    print('\nResults')
    print_10x10(sess, env, X_adv, name=args.dataset, attack='YY')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['mnist', 'cifar10', 'cifar100'], default='mnist')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-m', '--model', choices=['cnn_1', 'cnn_2', 'vgg16'], default='cnn_1')
    args = parser.parse_args()

    args.targeted = True

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
