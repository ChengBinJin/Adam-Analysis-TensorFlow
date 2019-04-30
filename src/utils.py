import os
import csv
import matplotlib.pyplot as plt


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


def make_folders(is_train=True, base=None, cur_time=None):
    if is_train:
        model_dir = os.path.join(base, 'model', '{}'.format(cur_time))
        log_dir = os.path.join(base, 'logs', '{}'.format(cur_time))

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    else:
        model_dir = os.path.join(base, 'model', '{}'.format(cur_time))
        log_dir = os.path.join(base, 'logs', '{}'.format(cur_time))

    return model_dir, log_dir



class CSVWriter(object):
    def __init__(self, path, name):
        self.file = open(os.path.join(path, name) + '.csv', mode='w', newline='')
        self.writer = csv.writer(self.file, delimiter=',')

    def update(self, iter_time, loss):
        self.writer.writerow([iter_time, loss])

    def close(self):
        self.file.close()
