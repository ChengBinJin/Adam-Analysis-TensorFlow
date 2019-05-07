########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import pickle
import os
import download
import numpy as np
from dataset import one_hot_encoded
from utils import plot_images, init_logger


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.

def maybe_download_and_extract(data_path):
    # URL for the data-set on the internet.
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_path)

########################################################################


class CIFAR10(object):
    def __init__(self, data_dir="../../Data/cifar10/", log_dir=None, is_train=False):
        self.data_path = data_dir
        maybe_download_and_extract(data_path=self.data_path)

        # Various constants for the size of the images.
        # Use these constants in your own program.

        # Width and height of each image.
        self.img_size = 32

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self.num_channels = 3
        self.img_shape = (self.img_size, self.img_size, self.num_channels)

        # Length of an image when flattened to a 1-dim array.
        self.img_size_flat = self.img_size * self.img_size * self.num_channels

        # Number of classes.
        self.num_classes = 10

        ########################################################################
        # Various constants used to allocate arrays of the correct size.

        # Number of files for the training-set.
        self._num_files_train = 5

        # Number of images for each batch-file in the training-set.
        self._images_per_file = 10000

        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file

        # Read training, validation, and test data
        self._load_training_data()  # training and validation data
        self._load_test_data()      # test data
        self._load_class_names()    # class names

        # init logger
        self.logger, self.file_handler, self.stream_handler = init_logger(log_dir=log_dir,
                                                                          name='cifar10',
                                                                          is_train=is_train)
    def preprocessing(self, use_whiten=True):
        if use_whiten:
            self._whitening()
        else:
            self._subtract_mean()

    def _subtract_mean(self):
        # Data matrix X to size [N x D]
        x_train_2d = np.reshape(self.x_train, (self.x_train.shape[0], -1))
        x_val_2d = np.reshape(self.x_val, (self.x_val.shape[0], -1))
        x_test_2d = np.reshape(self.x_test, (self.x_test.shape[0], -1))

        mean = np.mean(x_train_2d, axis=0)
        self.x_train = np.reshape((x_train_2d - mean), (self.x_train.shape[0], *self.img_shape))
        self.x_val = np.reshape((x_val_2d - mean), (self.x_val.shape[0], *self.img_shape))
        self.x_test = np.reshape((x_test_2d - mean), (self.x_test.shape[0], *self.img_shape))

    def _whitening(self):
        self.x_train, mean, U, S = self.whiten_preprocessing(self.x_train)
        self.x_val = self.whiten_preprocessing(self.x_val, mean_val=mean, U_val=U, S_val=S)
        self.x_test = self.whiten_preprocessing(self.x_test, mean_val=mean, U_val=U, S_val=S)

    def whiten_preprocessing(self, X, mean_val=None, U_val=None, S_val=None):
        # Input data matrix X of size [N x D]
        X = np.reshape(X, (X.shape[0], -1))

        mean = None
        if (mean_val is None) or (U_val is None) or (S_val is None):
            mean = np.mean(X, axis=0)
            X -= mean  # zero-center the data (important)
            cov = np.dot(X.T, X) / X.shape[0]  # get the data covariance matrix
            U, S, V = np.linalg.svd(cov)
        else:
            X -= mean_val
            U, S = U_val, S_val

        Xrot = np.dot(X, U)  # decorrelate the data

        # Whiten the data:
        # Divide by eigenvalues (which are square roots of the singular values)
        Xwhite = Xrot / np.sqrt(S + 1e-5)

        # Reshape original shape
        Xwhite = np.reshape(Xwhite, (Xwhite.shape[0], *self.img_shape))

        if (mean_val is None) or (U_val is None) or (S_val is None):
            return Xwhite, mean, U, S
        else:
            return Xwhite

    def info(self, show_img=False, use_logging=True, smooth=True):
        if use_logging:
            self.logger.info("Size of:")
            self.logger.info("- Training-set:\t\t{}".format(self.num_train))
            self.logger.info("- Validation-set:\t\t{}".format(self.num_val))
            self.logger.info("- Test-set:\t\t{}".format(self.num_test))

            self.logger.info("- image_size_flat: \t{}".format(self.img_size_flat))
            self.logger.info("- image_size: \t\t{}".format(self.img_shape))
            self.logger.info("- num_classes:\t\t{}".format(self.num_classes))

            self.logger.info("- CIFAR-10 class names: {}".format(self.class_names))
        else:
            print("Size of:")
            print("- Training-set:\t\t{}".format(self.num_train))
            print("- Validation-set:\t{}".format(self.num_val))
            print("- Test-set:\t\t{}".format(self.num_test))

            print("- image_size_flat: \t{}".format(self.img_size_flat))
            print("- image_size: \t\t{}".format(self.images_train[0].shape))
            print("- num_classes:\t\t{}".format(self.num_classes))

            print("- CIFAR-10 class names: {}".format(self.class_names))

        if show_img:
            index = np.random.choice(self.num_test, size=9, replace=False)
            plot_images(images=self.x_test[index], cls_true=self.x_test_cls[index], dataset='cifar10',
                        class_names=self.class_names, smooth=smooth)

    def _load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        self.images_train = np.zeros(shape=[self._num_images_train, self.img_size, self.img_size, self.num_channels],
                                     dtype=float)
        self.cls_train = np.zeros(shape=[self._num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(self._num_files_train):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            self.images_train[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            self.cls_train[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        self.labels_train = one_hot_encoded(class_numbers=self.cls_train, num_classes=self.num_classes)

        # Split into validation data
        self.num_val = int(round(self._num_images_train * 0.2))
        self.x_val = self.images_train[-self.num_val:]
        self.y_val_cls = self.cls_train[-self.num_val:]
        self.y_val = self.labels_train[-self.num_val:]

        self.num_train = self._num_images_train - self.num_val
        self.x_train = self.images_train[:self._num_images_train]
        self.y_train_cls = self.cls_train[:self._num_images_train]
        self.y_train = self.labels_train[:self._num_images_train]

    def _load_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        self.x_test, self.x_test_cls = self._load_data(filename="test_batch")
        # self.x_test_ori = self.x_test.copy()
        self.y_test = one_hot_encoded(class_numbers=self.x_test_cls, num_classes=self.num_classes)
        self.num_test = len(self.x_test)

    def random_batch(self, batch_size=32):
        # Number of images in the training-set.
        num_images = len(self.images_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=batch_size,
                               replace=False)

        # Use the random index to select random images and labels.
        x_batch = self.images_train[idx, :, :, :]
        y_batch = self.labels_train[idx, :]
        y_batch_cls = self.cls_train[idx]

        return x_batch, y_batch, y_batch_cls

    def _load_data(self, filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, cls

    def _load_class_names(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """

        # Load the class-names from the pickled file.
        raw = self._unpickle(filename="batches.meta")[b'label_names']

        # Convert from binary strings.
        self.class_names = [x.decode('utf-8') for x in raw]

    def _unpickle(self, filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self._get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        return data

    def _get_file_path(self, filename=""):
        """
        Return the full path of a data-file for the data-set.
        If filename=="" then return the directory of the files.
        """
        return os.path.join(self.data_path, "cifar-10-batches-py/", filename)

    def _convert_images(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.num_channels, self.img_size, self.img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images
