import os
# import numpy as np
import tensorflow as tf

from mnist import MNIST
from utils import make_folders
from models import Logistic
from solver import Solver

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size: default: 128')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_integer('epochs', 3, 'number of epochs, default: 100')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('random_seed', 123, 'random seed for python')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190427-1109), default: None')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Evaluation optimizers and dropout
    optimizer_options = ['SGDNesterov', 'Adagrad', 'RMSProp', 'AdaDelta', 'Adam']
    dropout_options = [False, True]

    # Initialize dataset and print info
    data = MNIST()
    data.info()  # print basic information

    num_iters = int(round(FLAGS.epochs * data.num_train / FLAGS.batch_size))
    for optimizer in optimizer_options:
        for dropout in dropout_options:
            tf.set_random_seed(FLAGS.random_seed)  # Fix weight initialization of each model with different optimizers

            print('Optimizer: {}\tDropout option: {}'.format(optimizer, dropout))
            sess = tf.Session()  # Initialize session

            # Initialize model and log folders
            # model_dir, log_dir = make_folders(is_train=FLAGS.is_train,
            #                                   base='logistic',
            #                                   mode=optimizer + '_' + str(dropout),
            #                                   load_model=FLAGS.load_model)

            # Initialize model
            model = Logistic(input_dim=data.img_size_flat,
                             output_dim=1,
                             optimizer=optimizer,
                             use_dropout=dropout,
                             lr=FLAGS.learning_rate,
                             random_seed=FLAGS.random_seed,
                             name=optimizer + '_' + str(dropout))

            # Initialize solver
            solver = Solver(sess, model)
            sess.run(tf.global_variables_initializer())

            for i in range(num_iters):
                x_batch, _, y_batch_cls = data.random_batch(batch_size=FLAGS.batch_size)

                _, loss, lr = solver.train(x_batch, y_batch_cls)

                if i % FLAGS.print_freq == 0:
                    print('{0:7}/{1:7}: Loss: {2:.3f}, Learning rate: {3}'.format(i, num_iters, loss, lr))

            print('Finished!')
            sess.close()
            tf.reset_default_graph()  # To release GPU memory

if __name__ == '__main__':
    tf.app.run()
