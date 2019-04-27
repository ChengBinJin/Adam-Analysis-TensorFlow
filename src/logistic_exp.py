import os
import numpy as np
import tensorflow as tf

from mnist import MNIST
from utils import plot_images, make_folders
# from solver import Solver

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size: default: 128')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_integer('epochs', 10, 'number of epochs, default: 100')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190427-1109), default: None')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    optimizer_options = ['SGDNesterov', 'Adagrad', 'RMSProp', 'AdaDelta', 'Adam', 'AdaMax']
    dropout_options = [False, True]

    # tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.99, name='SGDNesterov', use_nesterov=True)
    # model = Model()
    # solver = Solver(model)

    data = MNIST()
    data.info()  # print basic information

    num_iters = int(round(FLAGS.epochs * data.num_train / FLAGS.batch_size))
    for optimizer in optimizer_options:
        print('Optimizer: {}'.format(optimizer))

        for dropout in dropout_options:
            print('Dropout option: {}'.format(dropout))

            model_dir, log_dir = make_folders(is_train=FLAGS.is_train,
                                              base='logistic',
                                              mode=optimizer + '_' + str(dropout),
                                              load_model=FLAGS.load_model)
            # solver.init()

            for i in range(num_iters):
                if i % FLAGS.print_freq == 0:
                    print('iter time: {}'.format(i))

if __name__ == '__main__':
    tf.app.run()
