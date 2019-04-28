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

        return self.sess.run([train_op, loss], feed_dict=feed)

    def evaluate(self, X, y, batch_size=None):
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        if batch_size:
            num_data = X.shape[0]
            total_loss, total_acc = 0., 0.

            i = 0
            while i < num_data:
                # The ending index for the next batch is denoted j.
                j = min(i + batch_size, num_data)

                feed = {
                    self.model.X: X[i:j, :],
                    self.model.y: y[i:j],
                    self.model.keep_prob: 1.0
                }

                loss = self.model.loss
                acc = self.model.accuracy
                step_loss, step_acc = self.sess.run([loss, acc], feed_dict=feed)

                total_loss += step_loss * batch_size
                total_acc += step_acc * batch_size
                i = j

            total_loss /= num_data
            total_acc /= num_data

            return total_loss, total_acc

        else:
            feed = {
                self.model.X: X,
                self.model.y: y,
                self.model.keep_prob: 1.0
            }

            loss = self.model.loss
            acc = self.model.accuracy

            total_loss, total_acc = self.sess.run([loss, acc], feed_dict=feed)

            return total_loss, total_acc
