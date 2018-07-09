from ops import *
from autoencoder import AutoEncoder
from rnn import RNN


class BeatGenerator:
    def __init__(
            self,
            training_data_dir,
            logdir,
            autoencoder_config,
            rnn_config,
            z_length=32,
            z_output_fn=normalize,
    ):

        self.feed = FeedDict(training_data_dir, logdir)
        self.autoencoder = AutoEncoder(
            z_length=z_length,
            z_output_fn=z_output_fn,
            **autoencoder_config
        )
        self.rnn = RNN(
            z_length=z_length,
            z_output_fn=z_output_fn,
            **rnn_config
        )

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def training_loop(self, network):
        def train(
                batch_size,
                steps,
                summary_interval=10,
                test_interval=1000,
                save_interval=1000
        ):
            self.feed.change_dataset(network.scope_name)

            for i in range(steps):
                feed_dict = self.feed.next_batch(batch_size)

                gs, loss, loss_sum, _ = self.sess.run(
                    [network.global_step, network.loss, network.loss_sum, network.train_op],
                    feed_dict
                )


    def train_autoencoder(self, batch_size, steps, summary_interval=10, test_interval=1000, save_interval=1000):
        for i in range(steps):
            self.sess.run()


    def train_rnn(self, batch_size, steps, summary_interval=10, test_interval=1000, save_interval=1000):
        pass

    def encode_spectograms(self, input, show_loss=False):
        feed_dict = {self.autoencoder.x_placeholder: input}
        if show_loss:
            z, loss = self.sess.run([self.autoencoder.z, self.autoencoder.loss], feed_dict)
            print('Autoencoder Loss: {}'.format(loss))
        else:
            z = self.sess.run(self.autoencoder.z, feed_dict)
        return z
