import tensorflow as tf
import numpy as np

weight_init = tf.keras.initializers.he_normal()
bias_init = tf.constant_initializer(0)

def avg_pool(input, k=2):
    return tf.nn.avg_pool(
        input,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME'
    )

def resize(input, dims=None):
    if dims is None:
        s = tf.shape(input)
        dims = (s[1] // 2, s[2] // 2)
    return tf.image.resize_nearest_neighbor(input, dims)

def conv2d(input, out_channels, filter_size=3, k=1, padding='SAME'):
    if len(input.get_shape()) == 3:
        input = tf.expand_dims(input, [3])
    filter = tf.get_variable('filter',
        [filter_size, filter_size, input.get_shape()[3], out_channels],
        initializer=weight_init)
    filter = filter * tf.sqrt(2 / (filter_size ** 2 * int(input.get_shape()[3])))
    b = tf.get_variable('bias', out_channels, initializer=bias_init)
    output = tf.nn.conv2d(input, filter, [1, k, k, 1], padding) + b
    if out_channels == 1:
        output = tf.squeeze(output, 3)
    return output


def conv2d_transpose(input, out_channels, filter_size=3, k=1, w_init=weight_init, b_init=bias_init):
    input_shape = tf.shape(input)
    output_shape = input_shape[:]
    output_shape[2:4] = [input_shape[2] * 2, input_shape[3] * 2]

    filter_shape = [filter_size, filter_size, out_channels, input_shape[1]]
    filter = tf.get_variable( 'filter', filter_shape, initializer=w_init)
    filter = filter * tf.sqrt(2 / (filter_size ** 2 * input_shape[1]))

    b = tf.get_variable('b', out_channels, initializer=b_init)

    output = tf.nn.conv2d_transpose(
        input, filter, output_shape, [1, 1, k, k], 'SAME', data_format='NCHW'
    ) + b
    return output


def dense(input, output_size, w_init=weight_init, b_init=bias_init):
    W = tf.get_variable('W', [tf.shape(input)[-1], output_size], initializer=w_init)
    W = W * tf.sqrt(2 / int(tf.shape(input)[-1]))
    b = tf.get_variable('b', output_size, initializer=b_init)
    return tf.matmul(input, W) + b

def leaky_relu(input, alpha=0.2):
    return tf.nn.leaky_relu(input, alpha=alpha)

def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, epsilon=1e-5)

def layer_norm(input):
    return tf.contrib.layers.layer_norm(input)

def dropout(input, kp=0.75):
    return tf.nn.dropout(input, keep_prob=kp)

def pixelwise_norm(input, epsilon=1e-8):
    return input * tf.rsqrt(tf.reduce_mean(tf.square(input), axis=1, keepdims=True) + epsilon)

def minibatch_stddev(input):
    shape = tf.shape(input)
    x_ = tf.tile(tf.reduce_mean(input, 0, keepdims=True), [shape[0], 1, 1, 1])
    sigma = tf.sqrt(tf.reduce_mean(tf.square(input - x_), 0, keepdims=True) + 1e-4)
    sigma_avg = tf.reduce_mean(sigma, keepdims=True)
    layer = tf.tile(sigma_avg, [shape[0], shape[1], shape[2], 1])
    return tf.concat((input, layer), 3)

def normalize(input):
    mean = tf.reduce_mean(input, 1, keepdims=True)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(input), axis=1, keep_dims=True) + 1e-8)
    return (input - mean) / stddev

def embedding_lookup(ids, embedding_shape):
    limit = np.sqrt(6 / np.prod(embedding_shape[1:]))
    initializer = tf.random_uniform_initializer(-limit, limit)
    embeddings = tf.get_variable('embedding_table', embedding_shape, initializer=initializer)
    return tf.nn.embedding_lookup(embeddings, ids)

def get_train_op(network):
    with tf.variable_scope('{}_Optimizer'.format(network)):

        var_list = tf.trainable_variables()
        var_list = [var for var in var_list if network.scope_name in var.name]

        optimizer = tf.train.AdamOptimizer(**network.optimizer_config)
        train_op = optimizer.minimize(network.loss, global_step=network.global_step, var_list=var_list)

    return train_op