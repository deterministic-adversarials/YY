import numpy as np
import tensorflow as tf


def model(x, args, logits=False, training=False):                           # (None, 32, 32, 3)

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 32, 32, 64)
        z = tf.layers.dropout(z, rate=0.3, training=training)
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 32, 32, 64)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)         # (None, 16, 16, 64)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 16, 16, 128)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 16, 16, 128)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)         # (None, 8, 8, 128)

    with tf.variable_scope('conv2'):
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 8, 8, 256)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 8, 8, 256)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=256, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 8, 8, 256)              
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)         # (None, 4, 4, 256)

    with tf.variable_scope('conv3'):
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 4, 4, 512)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 4, 4, 512)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 4, 4, 512)             
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)         # (None, 2, 2, 512)
    
    with tf.variable_scope('conv4'):
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 2, 2, 512)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 2, 2, 512)
        z = tf.layers.dropout(z, rate=0.4, training=training)
        z = tf.layers.conv2d(z, filters=512, kernel_size=[3, 3], strides=1,
                             padding='same', activation=tf.nn.relu)         # (None, 2, 2, 512)         
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)         # (None, 1, 1, 512)
        z = tf.layers.dropout(z, rate=0.5, training=training)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])                         # (None, 512)

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=512, activation=tf.nn.relu)            # (None, 512)
        z = tf.layers.dropout(z, rate=0.5, training=training)

    logits_ = tf.layers.dense(z, units=args.n_classes, name='logits')            # (None, n_classes)
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def vgg16(args):

    class Dummy:
        pass

    env = Dummy()

    with tf.variable_scope('model'):
        env.x = tf.placeholder(tf.float32, (None, args.img_size, args.img_size, args.img_chan),
                            name='x')
        env.y = tf.placeholder(tf.float32, (None, args.n_classes), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')

        env.ybar, logits = model(env.x, args, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                        logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9) # SGD+momentum
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()
    
    return env