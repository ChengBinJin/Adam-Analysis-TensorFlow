import tensorflow as tf

class Logistic(object):
    def __init__(self, input_dim, output_dim=1, optimizer=None, use_dropout=True, lr=0.001, random_seed=123, name=None):
        with tf.variable_scope(name):
            tf.set_random_seed(random_seed)

            self.X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='X')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            net = self.X
            if use_dropout:
                net = tf.nn.dropout(x=net,
                                    keep_prob=self.keep_prob,
                                    seed=tf.set_random_seed(random_seed),
                                    name='dropout')

            self.y_pred = tf.layers.dense(inputs=net, units=output_dim, activation=None)
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.y_pred - self.y))
            self.train_op = optimizer_fn(optimizer, lr=lr, loss=self.loss)

            # Accuracy etc
            self.y_pred_round = tf.math.round(x=self.y_pred, name='rounded_pred')
            accuracy = tf.equal(tf.cast(x=self.y_pred_round, dtype=tf.int32), tf.cast(x=self.y, dtype=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(x=accuracy, dtype=tf.float32))


def optimizer_fn(optimizer, lr, loss):
    if optimizer == 'SGDNesterov':
        return tf.train.MomentumOptimizer(learning_rate=lr,
                                          momentum=0.99,
                                          name='SGDNesterov',
                                          use_nesterov=True).minimize(loss)
    elif optimizer == 'Adagrad':
        return tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss)

    elif optimizer == 'RMSProp':
        return tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

    elif optimizer == 'AdaDelta':
        return tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss)

    elif optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    elif optimizer == 'AdaMax':
        return tf.keras.optimizers.Adamax(lr=lr).minimize(loss)
