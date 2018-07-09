from ops import *


class AutoEncoder:
    def __init__(
            self,
            training_data_dir,
            input_d1,
            input_d2,
            z_length,
            beta=1.0,
            vae=False,
            channels=None,
            norm_fn=pixelwise_norm,
            activation_fn=leaky_relu,
            z_output_fn=normalize,
            x_output_fn=tf.tanh,
            optimizer_learning_rate=0.001,
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            scope_name='Autoencoder'
    ):

        self.feed = FeedDict(training_data_dir)

        self.channels = channels if channels else [512, 256, 128, 64, 32, 16, 8]
        self.n_layers = len(self.channels)
        self.x_d1 = input_d1
        self.x_d2 = input_d2

        reduction = 2 ** self.n_layers
        assert self.x_d1 % reduction == 0 and self.x_d2 % reduction == 0
        self.conv_output_shape = [
            self.channels[-1],
            self.x_d1 // reduction,
            self.x_d2 // reduction
        ]

        self.norm_fn = norm_fn
        self.hidden_activation_fn = activation_fn
        self.encoder_output_fn = z_output_fn
        self.decoder_output_fn = x_output_fn
        self.z_length = z_length
        self.vae = vae
        self.beta = beta

        self.optimizer_config = {
            'learning_rate': optimizer_learning_rate,
            'beta1': optimizer_beta1,
            'beta2': optimizer_beta2
        }
        self.scope_name = scope_name

        with tf.variable_scope(self.scope_name):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.x_placeholder = tf.placeholder(tf.float32)

            self.z, mu, sigma = self.encoder(self.x_placeholder)
            self.x_hat = self.decoder(self.z)
            self.loss = self.loss(self.x_placeholder, self.x_hat, mu, sigma)
            self.loss_sum = tf.summary.scalar('Loss', self.loss)
            self.train_op = get_train_op(self)


    def block(self, linear_fn):

        def block_fn(input, channel_idx, **fn_params):
            out_channels = self.channels[channel_idx]
            output = linear_fn(input, out_channels=out_channels, **fn_params)
            output = self.hidden_activation_fn(self.norm_fn(output))
            return output

        return block_fn


    def encoder(self, x):

        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            x = tf.expand_dims(x, 1)

            for i in range(self.n_layers - 1):
                with tf.variable_scope('Conv_Layer_{}'.format(i)):

                    with tf.variable_scope('0'):
                        x = self.block(conv2d)(x, i)

                    with tf.variable_scope('1'):
                        x = self.block(conv2d)(x, i, k=2)

            with tf.variable_scope('1x1_Conv'):

                x = self.block(conv2d)(x, -1, filter_size=1)
                x_flat_shape = [tf.shape(x)[0], np.prod(self.conv_output_shape)]
                x = tf.reshape(x, x_flat_shape)

            if self.vae:
                with tf.variable_scope('Reparameterization'):

                    with tf.variable_scope('mu'):
                        mu = dense(x, self.z_length)

                    with tf.variable_scope('sigma'):
                        sigma = dense(x, self.z_length)

                    epsilon = tf.random_normal([tf.shape(sigma)[0], self.z_length])
                    x = mu + tf.exp(sigma) * epsilon
                    z = self.encoder_output_fn(x)

                return z, mu, sigma

            else:
                with tf.variable_scope('Dense'):
                    x = dense(x, self.z_length)
                    z = self.encoder_output_fn(x)

                return z, None, None


    def decoder(self, z):

        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):

            with tf.variable_scope('Dense'):
                z = self.block(dense)(z, self.channels[-1])
                z_matrix_shape = [tf.shape(z)[0], *self.conv_output_shape]
                z = tf.reshape(z, z_matrix_shape)

            with tf.variable_scope('1x1_Conv'):
                z = self.block(conv2d)(z, self.channels[-2], filter_size=1)

            for i in reversed(range(self.n_layers - 1)):
                with tf.variable_scope('Conv_Layer_{}'.format(i)):

                    with tf.variable_scope('0'):
                        z = self.block(conv2d_transpose)(z, i, k=2)

                    with tf.variable_scope('1'):
                        if i  >= 1:
                            z = self.block(conv2d_transpose)(z, i - 1)

                        else:
                            z = conv2d_transpose(z, 1)
                            z = tf.squeeze(z, 1)

            x_hat = self.decoder_output_fn(z)

        return x_hat


    def loss(self, x, x_hat, mu=None, sigma=None):

        with tf.variable_scope('Loss_Function'):
            reconstruction_loss = x * -tf.log(x_hat) + (1 - x) * tf.log(1 - x)
            total_loss = tf.reduce_sum(reconstruction_loss, (1, 2))

            if mu is not None and sigma is not None:
                KL_divergence = 1 + 2 * sigma - mu ** 2 - tf.exp(2 * sigma)
                total_loss += -0.5 * self.beta * tf.reduce_sum(KL_divergence, (1, 2))

            total_loss = tf.reduce_mean(total_loss)
        return total_loss