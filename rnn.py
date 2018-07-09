from ops import *


class RNN:
    def __init__(
            self,
            z_length=32,
            hidden_layers=None,
            norm_fn=pixelwise_norm,
            dense_activation_fn=leaky_relu,
            z_output_fn=tf.sigmoid,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            scope_name='RNN_Network'
    ):

        self.z_length = z_length
        self.hidden_layers = hidden_layers if hidden_layers else [64]
        self.hidden_layers.append(z_length)
        self.n_layers = len(hidden_layers)
        self.norm_fn = norm_fn
        self.dense_activation_fn = dense_activation_fn
        self.output_fn = z_output_fn
        self.optimizer_config = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2
        }
        self.scope_name = scope_name

        with tf.variable_scope(self.scope_name):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.rnn_cell = tf.contrib.rnn.GRUCell(self.hidden_layers[0])
            self.z_placeholder = tf.placeholder(tf.float32)
            self.z_hat = self.training_rnn(self.z_placeholder)
            self.rnn_loss = self.loss(self.z_placeholder, self.z_hat)
            self.rnn_loss_sum = tf.summary.scalar('Loss', self.rnn_loss)
            self.train_op = get_train_op(self)


    def training_rnn(self, z):

        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            z = tf.nn.dynamic_rnn(self.rnn_cell, z)
            z_hat = self.__dense_layers(z)

        return z_hat


    def generating_rnn(self, z):
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            pass


    def __dense_layers(self, z):

        for i in range(1, self.n_layers):
            with tf.variable_scope('Dense_{}'.format(i)):

                n_hidden = self.hidden_layers[i]
                z = dense(z, n_hidden)

                if i == self.n_layers - 1:
                    z = self.output_fn(z)

                else:
                    z = self.dense_activation_fn(z)
                    z = self.norm_fn(z)

        return z


    def loss(self, z, z_hat):
        with tf.variable_scope('Loss_Function'):
            z = z[1:, :]
            z_hat = z_hat[:-1, :]
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_hat, labels=z
            )
            loss = tf.reduce_sum(loss, 1)
            loss = tf.reduce_mean(loss)
        return loss