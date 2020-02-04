import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .preparation_datasets import *
from .utils import predict


##
def out_(sess, env, X_adv, X_test, y_test, name='mnist', attack='YY'):

    if name=='mnist':
        img_size = 28
        img_chan = 1
        n_classes = 10
        X_tmp = np.empty((n_classes, img_size, img_size))
    elif name=='cifar10':
        img_size = 32
        img_chan = 3
        n_classes = 10
        X_tmp = np.empty((n_classes, img_size, img_size, img_chan))
    

    print('\nRandomly sample adversarial data from each category')

    y1 = predict(sess, env, X_test)
    y2 = predict(sess, env, X_adv)

    z0 = np.argmax(y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)

    y_tmp = np.empty((n_classes, n_classes))
    for i in range(n_classes):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        cur = np.random.choice(ind)
        X_tmp[i] = np.squeeze(X_adv[cur])
        y_tmp[i] = y2[cur]


    print('\nPlotting results')

    fig = plt.figure(figsize=(n_classes, 1.2))
    gs = gridspec.GridSpec(1, n_classes, wspace=0.05, hspace=0.05)

    label = np.argmax(y_tmp, axis=1)
    proba = np.max(y_tmp, axis=1)
    for i in range(n_classes):
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
def sample_img(sess, env, X_adv, X_test, y_test, name='mnist'):

    if name=='mnist':
        img_size = 28
        img_chan = 1
        n_classes = 10
    elif name=='cifar10':
        img_size = 32
        img_chan = 3
        n_classes = 10


    y_test_label = np.argmax(y_test[0:10], axis=1)
    y_test_proba = np.max(y_test, axis=1)
    
    y_adv = predict(sess, env, X_adv[0:10])
    y_adv_label = np.argmax(y_adv, axis=1)
    y_adv_proba = np.max(y_adv, axis=1)
    
    fig = plt.figure(figsize=(n_classes, 2.2))
    gs = gridspec.GridSpec(2, n_classes, wspace=0.05, hspace=0.05)
    
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
