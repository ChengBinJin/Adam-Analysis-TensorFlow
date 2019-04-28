import os
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

def plot_images(images, cls_true, dataset="mnist"):
    assert len(images) == len(cls_true) == 9

    image_shape = (28, 28)
    if dataset == "cifar10":
        image_shape = (32, 32, 3)
    elif dataset == "mnist":
        image_shape = image_shape
    else:
        raise NotImplementedError("Dataset not considered!")

    # Create figure with 3x3 sub-plots.
    plt.ion()  # Turn the interactive mode on.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(image_shape), cmap='binary')

        # Show true classes
        xlabel = "True: {}".format(cls_true[i])
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots in a single Notebook cell.
    plt.show()
    plt.pause(2)  # Pause 2 seconds


def make_folders(is_train=True, base=None, mode=None, load_model=None):
    if is_train:
        if load_model is None:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            "%Y%m%d-%H%M"
            model_dir = os.path.join(base, '{}'.format(cur_time), mode, 'model')
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        else:
            cur_time = load_model
            model_dir = os.path.join(base, '{}'.format(cur_time), mode, 'model')

        log_dir = os.path.join(base, '{}'.format(cur_time), mode, 'logs').format(cur_time)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

    else:
        model_dir = os.path.join(base, '{}'.format(load_model), mode, 'model')
        log_dir = os.path.join(base, '{}'.format(load_model), mode, 'logs')

    return model_dir, log_dir


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


class Logistic(object):
    def __init__(self, input_dim, output_dim, optimizer=None, use_dropout=True, lr=0.001, random_seed=123, name=None):
        with tf.variable_scope(name):
            tf.set_random_seed(random_seed)

            self.X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='X')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            net = self.X
            if use_dropout:
                net = tf.nn.dropout(x=net, keep_prob=self.keep_prob, seed=tf.set_random_seed(seed), name='dropout')

            self.y_pred = tf.layers.dense(inputs=net, unit=output_dim, activation=None)
            self.loss = tf.nn.l2_loss(self.y_pred - self.y)
            self.optim = optimizer_fn(optimizer, lr=lr, loss=self.loss)

            # Accuracy etc
            self.y_pred_round = tf.math.round(x=self.y_pred, name='rounded_pred')
            self.accuracy = tf.equal(tf.cast(x=self.y_pred_round, dtype=tf.int32), tf.cast(x=self.y, dtype=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(x=self.accuracy, dtype=tf.float32))



