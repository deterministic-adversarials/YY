import os
import numpy as np
import random


## evaluate models
def evaluate(sess, env, X_data, y_data, batch_size=128):

    print('\nEvaluating')
    
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


## predict labels
def predict(sess, env, X_data, batch_size=128):

    print('\nPredicting')

    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


## train models
def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='default'):
    
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'models/{0}/{1}'.format(name,name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        lr = 0.1
        if epoch > 150:
            lr = 0.01
        if epoch > 250:
            lr = 0.001

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.learning_rate: lr,
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)
    
    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('models/{0}'.format(name), exist_ok=True)
        env.saver.save(sess, 'models/{0}/{1}'.format(name,name))


##
def pseudorandom_target(index, num_classes, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, num_classes)
    return target


## exclude misclassified samples in X_data
def exclude_miss(sess, env, X_data, y_data, first, last):
    """
    z0 = np.argmax(y_data, axis=1)
    z1 = np.argmax(predict(sess, env, X_data), axis=1)
    ind = z0 == z1

    X_data = X_data[ind]
    labels = z0[ind]
    """
    z0 = np.argmax(y_data[first:last], axis=1)
    z1 = np.argmax(predict(sess, env, X_data[first:last]), axis=1)
    miss_indices = np.where(z0 != z1)
    X = np.delete(X_data[first:last], miss_indices, axis=0)
    y = np.delete(y_data[first:last], miss_indices, axis=0)

    return (X, y)
