import tensorflow as tf

import tensorflow_utils as tf_utils
import utils as utils


def optimizer_fn(optimizer, lr, loss, name='Optimizer'):
    with tf.variable_scope(name):
        global_step = tf.Variable(1, dtype=tf.float32, trainable=False)
        cur_lr = lr / tf.math.sqrt(x=global_step)

        if optimizer == 'SGDNesterov':
            return tf.train.MomentumOptimizer(learning_rate=cur_lr,
                                              momentum=0.99,
                                              name='SGDNesterov',
                                              use_nesterov=True).minimize(loss, global_step=global_step), cur_lr
        elif optimizer == 'Adagrad':
            return tf.train.AdagradOptimizer(learning_rate=cur_lr).minimize(loss, global_step=global_step), cur_lr
        elif optimizer == 'RMSProp':
            return tf.train.RMSPropOptimizer(learning_rate=cur_lr).minimize(loss, global_step=global_step), cur_lr
        elif optimizer == 'AdaDelta':
            return tf.train.AdadeltaOptimizer(learning_rate=cur_lr).minimize(loss, global_step=global_step), cur_lr
        elif optimizer == 'Adam':
            return tf.train.AdamOptimizer(learning_rate=cur_lr).minimize(loss, global_step=global_step), cur_lr
        else:
            raise NotImplementedError(" [*] Optimizer is not included in list!")
        # elif optimizer == 'AdaMax':
            # return tf.contrib.opt.AdaMaxOptimizer(learning_rate=self.cur_lr).minimize(loss, global_step=global_step), cur_lr



class Logistic(object):
    def __init__(self, input_dim, output_dim=1, optimizer=None, use_dropout=True, lr=0.001, random_seed=123,
                 is_train=True, log_dir=None, name=None):
        self.name = name
        self.is_train = is_train
        self.log_dir = log_dir
        self.cur_lr = None
        self.logger, self.file_handler, self.stream_handler = utils.init_logger(log_dir=self.log_dir,
                                                                                name=self.name,
                                                                                is_train=self.is_train)
        with tf.variable_scope(self.name):
            # Placeholders for inputs
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='X')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            tf_utils.print_activations(self.X, logger=self.logger if self.is_train else None)

            # Placeholders for TensorBoard
            self.train_acc = tf.placeholder(tf.float32, name='train_acc')
            self.val_acc = tf.placeholder(tf.float32, name='val_acc')

            net = self.X
            if use_dropout:
                net = tf_utils.dropout(x=net,
                                       keep_prob=self.keep_prob,
                                       seed=random_seed,
                                       name='dropout',
                                       logger=self.logger if self.is_train else None)

            # Network, loss, and optimizer
            self.y_pred = tf_utils.linear(net, output_size=output_dim)
            tf_utils.print_activations(self.y_pred, logger=self.logger if self.is_train else None)
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.y_pred - self.y))
            self.train_op, self.cur_lr = optimizer_fn(optimizer, lr=lr, loss=self.loss, name=self.name)

            # Accuracy etc
            self.y_pred_round = tf.math.round(x=self.y_pred, name='rounded_pred')
            accuracy = tf.equal(tf.cast(x=self.y_pred_round, dtype=tf.int32), tf.cast(x=self.y, dtype=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(x=accuracy, dtype=tf.float32)) * 100.

        self._tensorboard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _tensorboard(self):
        self.summary_op = tf.summary.merge(inputs=[tf.summary.scalar('Loss', self.loss),
                                                   tf.summary.scalar('Learning_rate', self.cur_lr)])
        self.train_acc_op = tf.summary.scalar('Acc/train', self.train_acc)
        self.val_acc_op = tf.summary.scalar('Acc/val', self.val_acc)

    def release_handles(self):
        utils.release_handles(self.logger, self.file_handler, self.stream_handler)


class NeuralNetwork(object):
    def __init__(self, input_dim, output_dim=[1000, 1000, 10], optimizer=None, use_dropout=True, lr=0.001,
                 weight_decay=1e-4, random_seed=123, is_train=True, log_dir=None, name=None):
        self.name = name
        self.is_train = is_train
        self.log_dir = log_dir
        self.cur_lr = None
        self.logger, self.file_handler, self.stream_handler = utils.init_logger(log_dir=self.log_dir,
                                                                                name=self.name,
                                                                                is_train=self.is_train)

        with tf.variable_scope(self.name):
            # Placeholders for inputs
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='X')
            tf_utils.print_activations(self.X, logger=self.logger if self.is_train else None)
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim[-1]], name='y')
            self.y_cls = tf.math.argmax(input=self.y, axis=1)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # Placeholders for TensorBoard
            self.train_acc = tf.placeholder(tf.float32, name='train_acc')
            self.val_acc = tf.placeholder(tf.float32, name='val_acc')

            net = self.X
            for idx in range(len(output_dim) - 1):
                net = tf_utils.linear(x=net,
                                      output_size=output_dim[idx],
                                      name='fc'+str(idx),
                                      logger=self.logger if self.is_train else None)

                if use_dropout:
                    net = tf_utils.dropout(x=net,
                                           keep_prob=self.keep_prob,
                                           seed=random_seed,
                                           name='dropout'+str(idx),
                                           logger=self.logger if self.is_train else None)

                net = tf_utils.relu(x=net,
                                    name='relu'+str(idx),
                                    logger=self.logger if self.is_train else None)

            # Last predict layer
            self.y_pred = tf_utils.linear(net, output_size=output_dim[-1], name='last_fc')
            tf_utils.print_activations(self.y_pred, logger=self.logger if self.is_train else None)

            # Loss = data loss + regularization term
            self.data_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
            self.reg_term = weight_decay * tf.reduce_sum(
                [tf.nn.l2_loss(weight) for weight in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.loss = self.data_loss + self.reg_term

            # Optimizer
            self.train_op, self.cur_lr = optimizer_fn(optimizer, lr=lr, loss=self.loss, name=self.name)

            # Accuracy etc
            self.y_pred_cls = tf.math.argmax(input=self.y_pred, axis=1)
            correct_prediction = tf.math.equal(self.y_pred_cls, self.y_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32)) * 100.

        self._tensorboard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _tensorboard(self):
        self.summary_op = tf.summary.merge(inputs=[tf.summary.scalar('Loss\Total', self.loss),
                                                   tf.summary.scalar('Loss\Data', self.data_loss),
                                                   tf.summary.scalar('Loss\Reg', self.reg_term),
                                                   tf.summary.scalar('Learning_rate', self.cur_lr)])

        self.train_acc_op = tf.summary.scalar('Acc/train', self.train_acc)
        self.val_acc_op = tf.summary.scalar('Acc/val', self.val_acc)

    def release_handles(self):
        utils.release_handles(self.logger, self.file_handler, self.stream_handler)


    class CNN(object):
        def __init__(self, input_dim=[32, 32, 3], output_dim=[128, 256, 512, 1000, 10], optimizer=None, use_dropout=True, lr=0.001,
                     weight_decay=1e-4, random_seed=123, is_train=True, log_dir=None, name=None):
            self.name = name
            self.is_train = is_train
            self.log_dir = log_dir
            self.cur_lr = None
            self.logger, self.file_handler, self.stream_handler = utils.init_logger(log_dir=self.log_dir,
                                                                                    name=self.name,
                                                                                    is_train=self.is_train)

            with tf.variable_scope(self.name):
                # Placeholders for inputs
                self.X = tf.placeholder(dtype=tf.float32, shape=[None, *input_dim], name='X')
                self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim[-1]], name='y')
                self.y_cls = tf.math.argmax(input=self.y, axis=1)
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                tf_utils.print_activations(self.X, logger=self.logger if self.is_train else None)

                # Placeholders for TensorBoard
                self.train_acc = tf.placeholder(tf.float32, name='train_acc')
                self.val_acc = tf.placeholder(tf.float32, name='val_acc')

                # Convolutional layers
                net = self.X
                if use_dropout:
                    net = tf_utils.dropout(x=net,
                                           keep_prob=self.keep_prob,
                                           seed=random_seed,
                                           name='dropout_input',
                                           logger=self.logger if self.is_train else None)

                for idx in range(3):
                    net = tf_utils.conv2d(x=net,
                                          output_dim=output_dim[idx],
                                          k_h=5,
                                          k_w=5,
                                          d_h=1,
                                          d_w=1,
                                          name='conv2d'+str(idx),
                                          logger=self.logger if self.is_train else None)
                    net = tf_utils.max_pool(x=net,
                                            ksize=[1, 3, 3, 1],
                                            strides=[1, 2, 2, 1],
                                            name='maxpool'+str(idx),
                                            logger=self.logger if self.is_train else None)
                    net = tf_utils.relu(x=net,
                                        name='relu'+str(idx),
                                        is_print=True,
                                        logger=self.logger if self.is_train else None)

                # Fully conneted layers
                net = tf_utils.linear(x=net,
                                      output_size=output_dim[-2],
                                      name='fc3',
                                      logger=self.logger if self.is_train else None)

                if use_dropout:
                    net = tf_utils.dropout(x=net,
                                           keep_prob=self.keep_prob,
                                           seed=random_seed,
                                           name='dropout3',
                                           logger=self.logger if self.is_train else None)

                net = tf_utils.relu(x=net,
                                    name='relu3',
                                    logger=self.logger if self.is_train else None)

                # Last predict layer
                self.y_pred = tf_utils.linear(x=net,
                                              output_size=output_dim[-1],
                                              name='last_fc',
                                              logger=self.logger if self.is_train else None)

                # Loss = data loss + regularization term
                self.data_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
                self.reg_term = weight_decay * tf.reduce_sum(
                    [tf.nn.l2_loss(weight) for weight in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
                self.loss = self.data_loss + self.reg_term

                # Optimizer
                self.train_op, self.cur_lr = optimizer_fn(optimizer, lr=lr, loss=self.loss, name=self.name)

                # Accuracy etc
                self.y_pred_cls = tf.math.argmax(input=self.y_pred, axis=1)
                correct_prediction = tf.math.equal(self.y_pred_cls, self.y_cls)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32)) * 100.

            self._tensorboard()
            tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

        def _tensorboard(self):
            self.summary_op = tf.summary.merge(inputs=[tf.summary.scalar('Loss\Total', self.loss),
                                                       tf.summary.scalar('Loss\Data', self.data_loss),
                                                       tf.summary.scalar('Loss\Reg', self.reg_term),
                                                       tf.summary.scalar('Learning_rate', self.cur_lr)])

            self.train_acc_op = tf.summary.scalar('Acc/train', self.train_acc)
            self.val_acc_op = tf.summary.scalar('Acc/val', self.val_acc)

        def release_handles(self):
            utils.release_handles(self.logger, self.file_handler, self.stream_handler)
