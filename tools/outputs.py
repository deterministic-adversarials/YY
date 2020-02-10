import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .datasets import load_mnist, load_cifar10, load_cifar100
from .utils import predict


##
def print_out(args, sess, env, X_adv, X_test, y_test, name='mnist', attack='YY'):

    print('\nRandomly sample adversarial data from each category')

    y1 = predict(sess, env, X_test)
    y2 = predict(sess, env, X_adv)

    z0 = np.argmax(y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)

    X_tmp = np.empty((args.n_classes, args.img_size, args.img_size, args.img_chan))
    X_tmp = np.squeeze(X_tmp)
    y_tmp = np.empty((args.n_classes, args.n_classes))
    for i in range(args.n_classes):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        cur = np.random.choice(ind)
        X_tmp[i] = np.squeeze(X_adv[cur])
        y_tmp[i] = y2[cur]

    print('\nPlotting results')

    fig = plt.figure(figsize=(args.n_classes, 1.2))
    gs = gridspec.GridSpec(1, args.n_classes, wspace=0.05, hspace=0.05)

    label = np.argmax(y_tmp, axis=1)
    proba = np.max(y_tmp, axis=1)
    for i in range(args.n_classes):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                    fontsize=12)

    print('\nSaving figure')

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/{0}_{1}.png'.format(name, attack))


##
def print_sample(args, sess, env, X_adv, X_test, y_test):

    y_test_label = np.argmax(y_test[0:10], axis=1)
    y_test_proba = np.max(y_test, axis=1)
    
    y_adv = predict(sess, env, X_adv[0:10])
    y_adv_label = np.argmax(y_adv, axis=1)
    y_adv_proba = np.max(y_adv, axis=1)
    
    fig = plt.figure(figsize=(args.n_classes, 2.2))
    gs = gridspec.GridSpec(2, args.n_classes, wspace=0.05, hspace=0.05)
    
    # for image
    X_test = np.squeeze(X_test[0:10])
    X_adv = np.squeeze(X_adv[0:10])
    for i in range(10):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_test[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(y_test_label[i], y_test_proba[i]),
                    fontsize=12)

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(X_adv[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(y_adv_label[i], y_adv_proba[i]),
                    fontsize=12)

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/sample.png')


##
def print_10x10(args, sess, env, X_adv, name='mnist', attack='YY'):

    print('\nGenerating figure')

    fig = plt.figure(figsize=(args.n_classes, args.n_classes))
    gs = gridspec.GridSpec(args.n_classes, args.n_classes, wspace=0.1, hspace=0.1)

    for i in range(args.n_classes):
        for j in range(args.n_classes):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(X_adv[i, j], cmap='gray', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])

            if i == j:
                for spine in ax.spines:
                    ax.spines[spine].set_color('green')
                    ax.spines[spine].set_linewidth(5)

            if ax.is_first_col():
                ax.set_ylabel(i, fontsize=20, rotation='horizontal', ha='right')
            if ax.is_last_row():
                ax.set_xlabel(j, fontsize=20)

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/{0}_{1}_10x10.png'.format(name, attack))

