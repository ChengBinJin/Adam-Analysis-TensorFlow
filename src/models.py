import tensorflow as tf

class Logistic(object):
    def __init__(self, input_dim, output_dim=1, optimizer=None, use_dropout=True, lr=0.001, random_seed=123, name=None):
        self.name = name
        self.cur_lr = None

        with tf.variable_scope(self.name):
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
            self.train_op = self.optimizer_fn(optimizer, lr=lr, loss=self.loss, name=self.name)

            # Accuracy etc
            self.y_pred_round = tf.math.round(x=self.y_pred, name='rounded_pred')
            accuracy = tf.equal(tf.cast(x=self.y_pred_round, dtype=tf.int32), tf.cast(x=self.y, dtype=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(x=accuracy, dtype=tf.float32))

        self._tensorboard()


    def optimizer_fn(self, optimizer, lr, loss, name='Optimizer'):
        with tf.variable_scope(name):
            global_step = tf.Variable(1, dtype=tf.float32, trainable=False)
            self.cur_lr = lr / tf.math.sqrt(x=global_step)

            if optimizer == 'SGDNesterov':
                return tf.train.MomentumOptimizer(learning_rate=self.cur_lr,
                                                  momentum=0.99,
                                                  name='SGDNesterov',
                                                  use_nesterov=True).minimize(loss, global_step=global_step)
            elif optimizer == 'Adagrad':
                return tf.train.AdagradOptimizer(learning_rate=self.cur_lr).minimize(loss, global_step=global_step)

            elif optimizer == 'RMSProp':
                return tf.train.RMSPropOptimizer(learning_rate=self.cur_lr).minimize(loss, global_step=global_step)

            elif optimizer == 'AdaDelta':
                return tf.train.AdadeltaOptimizer(learning_rate=self.cur_lr).minimize(loss, global_step=global_step)

            elif optimizer == 'Adam':
                return tf.train.AdamOptimizer(learning_rate=self.cur_lr).minimize(loss, global_step=global_step)
            else:
                raise NotImplementedError(" [*] Optimizer is not included in list!")

            # elif optimizer == 'AdaMax':
                # return tf.contrib.opt.AdaMaxOptimizer(learning_rate=self.cur_lr).minimize(loss, global_step=global_step)

    def _tensorboard(self):
        self.summary_op = tf.summary.merge(inputs=[tf.summary.scalar('Loss', self.loss),
                                                   tf.summary.scalar('Learning_rate', self.cur_lr)])
