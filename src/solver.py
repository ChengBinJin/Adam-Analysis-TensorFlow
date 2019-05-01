import numpy as np


class Solver(object):
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model

    def train(self, x, y):
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        feed = {
            self.model.X: x,
            self.model.y: y,
            self.model.keep_prob: 0.5
        }
        train_op = self.model.train_op
        loss = self.model.loss
        summary = self.model.summary_op

        return self.sess.run([train_op, loss, summary], feed_dict=feed)

    def evaluate(self, X, y, batch_size=None, is_train=False):
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        if batch_size:
            num_data = X.shape[0]
            total_acc = 0.

            i = 0
            while i < num_data:
                # The ending index for the next batch is denoted j.
                j = min(i + batch_size, num_data)

                feed = {
                    self.model.X: X[i:j, :],
                    self.model.y: y[i:j],
                    self.model.keep_prob: 1.0
                }

                step_acc = self.sess.run(self.model.accuracy, feed_dict=feed)
                total_acc += step_acc * batch_size
                i = j

            total_acc /= num_data
        else:
            feed = {
                self.model.X: X,
                self.model.y: y,
                self.model.keep_prob: 1.0
            }

            total_acc = self.sess.run(self.model.accuracy, feed_dict=feed)

        if is_train:
            summary = self.sess.run(self.model.train_acc_op, feed_dict={self.model.train_acc: total_acc})
        else:
            summary = self.sess.run(self.model.val_acc_op, feed_dict={self.model.val_acc: total_acc})

        return total_acc, summary
