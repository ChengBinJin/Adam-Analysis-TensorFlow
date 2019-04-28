import os
import matplotlib.pyplot as plt
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
