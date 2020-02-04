import numpy as np
import tensorflow as tf


def model(x, logits=False, training=False):
    
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=512, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.5, training=training)
        z = tf.layers.dense(z, units=256, activation=tf.nn.relu)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def model_mnist():

    class Dummy:
        pass

    env = Dummy()

    with tf.variable_scope('model'):
        env.x = tf.placeholder(tf.float32, (None, 28, 28, 1),
                            name='x')
        env.y = tf.placeholder(tf.float32, (None, 10), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')

        env.ybar, logits = model(env.x, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                        logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.MomentumOptimizer(0.1,0.9) # SGD+momentum
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()
    
    return env
    

