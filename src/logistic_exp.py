import os
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

    if FLAGS.model.lower() == 'logistic':
        # Initialize dataset and print info
        data = MNIST()
        data.info()  # print basic information
    elif FLAGS.model.lower() == 'neural_network' or FLAGS.model.lower() == 'cnn':
        # Initialize dataset and print info
        data = MNIST()
        data.info()  # print basic information
    else:
        raise NotImplementedError

    if FLAGS.is_train:
        train(data, optimizer_options, dropout_options, model_dir, log_dir)
    else:
        test(data, optimizer_options, dropout_options, model_dir)

def train(data, optimizer_options, dropout_options, model_dir, log_dir):
    num_iters = int(round(FLAGS.epochs * data.num_train / FLAGS.batch_size))

    for optimizer in optimizer_options:
        for dropout in dropout_options:
            print('\nOptimizer: {}\tDropout option: {}\n'.format(optimizer, dropout))

            # Initialize sub folders for multiple models
            mode_name = optimizer + '_' + str(dropout)
            sub_model_dir = os.path.join(model_dir, mode_name)
            sub_log_dir = os.path.join(log_dir, mode_name)

            if not os.path.isdir(sub_model_dir):
                os.makedirs(sub_model_dir)

            if not os.path.isdir(sub_log_dir):
                os.makedirs(sub_log_dir)

            # Fix weight initialization of each model with different optimizers
            tf.set_random_seed(FLAGS.random_seed)
            sess = tf.Session()  # Initialize session

            # Initialize model
            model = Logistic(input_dim=data.img_size_flat,
                             output_dim=1,
                             optimizer=optimizer,
                             use_dropout=dropout,
                             lr=FLAGS.learning_rate,
                             random_seed=FLAGS.random_seed,
                             name=mode_name)

            # Initialize solver
            solver = Solver(sess, model)
            tb_writer = tf.summary.FileWriter(sub_log_dir, graph_def=solver.sess.graph_def)
            sess.run(tf.global_variables_initializer())

            csvWriter = CSVWriter(path=log_dir, name=mode_name)

            for iter_time in range(num_iters):
                x_batch, _, y_batch_cls = data.random_batch(batch_size=FLAGS.batch_size)
                _, loss, summary = solver.train(x_batch, y_batch_cls)
                csvWriter.update(iter_time, loss)

                # Write to tensorboard
                tb_writer.add_summary(summary, iter_time)
                tb_writer.flush()

                if iter_time % FLAGS.print_freq == 0:
                    print('{0:7}/{1:7}: Loss: {2:.3f}'.format(iter_time, num_iters, loss))

            csvWriter.close()
            sess.close()
            tf.reset_default_graph()  # To release GPU memory



def test(data, optimizer_options, dropout_options, model_dir):
    print('Hello test mode!')

if __name__ == '__main__':
    tf.app.run()
