import os
import csv
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

from mnist import MNIST
from utils import make_folders, CSVWriter
from models import Logistic
from solver import Solver

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('model', 'logistic', 'network model in [logistic|neural_network|cnn]')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size: default: 128')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for optimizer, default: 0.0001')
tf.flags.DEFINE_integer('epochs', 3, 'number of epochs, default: 100')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('random_seed', 123, 'random seed for python')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190427-1109), default: None')


logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def init_logger(log_dir):
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info('gpu_index: {}'.format(FLAGS.gpu_index))
    logger.info('model: {}'.format(FLAGS.model))
    logger.info('batch_size: {}'.format(FLAGS.batch_size))
    logger.info('is_train: {}'.format(FLAGS.is_train))
    logger.info('learning_rate: {}'.format(FLAGS.learning_rate))
    logger.info('epochs: {}'.format(FLAGS.epochs))
    logger.info('print_freq: {}'.format(FLAGS.print_freq))
    logger.info('random_seed: {}'.format(FLAGS.random_seed))
    logger.info('load_model: {}'.format(FLAGS.load_model))

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Evaluation optimizers and dropout
    optimizer_options = ['SGDNesterov', 'Adagrad', 'RMSProp', 'AdaDelta', 'Adam']
    dropout_options = [False, True]

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir = make_folders(is_train=FLAGS.is_train,
                                      base=FLAGS.model,
                                      cur_time=cur_time)
    init_logger(log_dir=log_dir)

    if FLAGS.model.lower() == 'logistic':
        # Initialize dataset and print info
        data = MNIST(log_dir=log_dir)
        data.info(use_logging=True if FLAGS.is_train else False)  # print basic information
    elif FLAGS.model.lower() == 'neural_network' or FLAGS.model.lower() == 'cnn':
        # Initialize dataset and print info
        data = MNIST(log_dir=log_dir)
        data.info(use_logging=True if FLAGS.is_train else False)  # print basic information
    else:
        raise NotImplementedError

    if FLAGS.is_train:
        train(data, optimizer_options, dropout_options, model_dir, log_dir)
    else:
        test(data, optimizer_options, dropout_options, model_dir)

def train(data, optimizer_options, dropout_options, model_dir, log_dir):
    num_iters = int(round(FLAGS.epochs * data.num_train / FLAGS.batch_size))
    iters_epoch = int(round(data.num_train / FLAGS.batch_size))

    # for optimizer in optimizer_options:
    #     for dropout in dropout_options:
    #         print('\nOptimizer: {}\tDropout option: {}\n'.format(optimizer, dropout))
    #
    #         # Initialize sub folders for multiple models
    #         mode_name = optimizer + '_' + str(dropout)
    #         sub_model_dir = os.path.join(model_dir, mode_name)
    #         sub_log_dir = os.path.join(log_dir, mode_name)
    #
    #         if not os.path.isdir(sub_model_dir):
    #             os.makedirs(sub_model_dir)
    #
    #         if not os.path.isdir(sub_log_dir):
    #             os.makedirs(sub_log_dir)
    #
    #         # Fix weight initialization of each model with different optimizers
    #         tf.set_random_seed(FLAGS.random_seed)
    #         sess = tf.Session()  # Initialize session
    #
    #         # Initialize model
    #         model = Logistic(input_dim=data.img_size_flat,
    #                          output_dim=1,
    #                          optimizer=optimizer,
    #                          use_dropout=dropout,
    #                          lr=FLAGS.learning_rate,
    #                          random_seed=FLAGS.random_seed,
    #                          is_train=FLAGS.is_train,
    #                          log_dir=sub_log_dir,
    #                          name=mode_name)
    #
    #         # Initialize solver
    #         solver = Solver(sess, model)
    #         saver = tf.train.Saver(max_to_keep=1)
    #         tb_writer = tf.summary.FileWriter(sub_log_dir, graph_def=solver.sess.graph_def)
    #         csvWriter = CSVWriter(path=log_dir, name=mode_name)
    #         sess.run(tf.global_variables_initializer())
    #
    #         best_acc = 0.
    #         for iter_time in range(num_iters):
    #             x_batch, _, y_batch_cls = data.random_batch(batch_size=FLAGS.batch_size)
    #             _, loss, summary = solver.train(x_batch, y_batch_cls)
    #             csvWriter.update(iter_time, loss)
    #
    #             # Write to tensorboard
    #             tb_writer.add_summary(summary, iter_time)
    #             tb_writer.flush()
    #
    #             if iter_time % FLAGS.print_freq == 0:
    #                 print('{0:7}/{1:7}: Loss: {2:.3f}'.format(iter_time, num_iters, loss))
    #
    #             # Validation
    #             if iter_time % iters_epoch == 0 or iter_time == (num_iters - 1):
    #                 acc = solver.evaluate(data.x_val, data.y_val_cls, batch_size=FLAGS.batch_size)
    #                 print('Iter: {0:7}, Acc: {1:.3f}, Best Acc: {2:.3f}'.format(iter_time, acc, best_acc))
    #
    #                 if acc > best_acc:
    #                     save_model(saver, solver, sub_model_dir, mode_name, iter_time)
    #                     best_acc = acc
    #
    #         model.release_handles()
    #         sess.close()
    #         tf.reset_default_graph()  # To release GPU memory
    #         csvWriter.close()

    plot_loss(log_dir, optimizer_options, dropout_options)

def test(data, optimizer_options, dropout_options, model_dir):
    print('Hello test mode!')

def save_model(saver, solver, sub_model_dir, mode_name, iter_time):
    saver.save(solver.sess, os.path.join(sub_model_dir, 'model'), global_step=iter_time)
    logger.info(' [*] Model saved! Mode: {}, Iter: {}'.format(mode_name, iter_time))


def plot_loss(log_dir, optimizer_options, dropout_options):
    optim_data, names = [], []

    # read csv files
    for optimizer in optimizer_options:
        for dropout in dropout_options:
            data = []
            names.append(optimizer + '_' + str(dropout))
            file_name = os.path.join(log_dir, optimizer + '_' + str(dropout) + '.csv')

            with open(file_name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    data.append(float(row[1]))

            optim_data.append(data)

    optim_data = np.asarray(np.transpose(optim_data))
    x = np.arange(optim_data.shape[0])

    sns.set()
    plt.rcParams['figure.figsize'] = (12.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'

    plt.plot(x, optim_data)
    plt.legend(names, ncol=2, loc='upper left')
    plt.title('Optimizers with dropout')
    plt.savefig(os.path.join(log_dir, FLAGS.model + '.png'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    tf.app.run()
