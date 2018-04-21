# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# This file is adapted from tensorflow official tutorial of mnist.
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf
from config import cfg
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

from CapsNet import CapsNet

FLAGS = None


def model_test():
    model = CapsNet(None)
    model.creat_architecture()
    print("pass")

# Taken from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
# ****************************************************
import numpy
import gzip
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile


DEFAULT_SOURCE_URL = "data/emnist/"

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
      ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.
    Returns:
      labels: a 1D uint8 numpy array.
    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    if FLAGS.dataset == "mnist" or FLAGS.dataset == "fashion-mnist" \
            or FLAGS.dataset == "emnist-digits":
        num_classes = 10
    elif FLAGS.dataset == "emnist-balanced" or FLAGS.dataset == "emnist-bymerge":
        num_classes = 47
    elif FLAGS.dataset == "emnist-letters":
        num_classes = 37
    elif FLAGS.dataset == "emnist-byclass":
        num_classes = 62

    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


class DataSet(object):
    """Container class for a dataset (deprecated).
    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.
    """

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate(
                (images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_data_sets_local(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
    if fake_data:
        def fake():
            return DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    if not source_url:  # empty string check
        source_url = DEFAULT_SOURCE_URL


    if FLAGS.dataset == "mnist":
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    elif FLAGS.dataset == "emnist-balanced":
        print("Reading emnist-balanced")
        TRAIN_IMAGES = 'emnist-balanced-train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'emnist-balanced-train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 'emnist-balanced-test-images-idx3-ubyte.gz'
        TEST_LABELS = 'emnist-balanced-test-labels-idx1-ubyte.gz'

    elif FLAGS.dataset == "emnist-byclass":
        TRAIN_IMAGES = 'emnist-byclass-train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'emnist-byclass-train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 'emnist-byclass-test-images-idx3-ubyte.gz'
        TEST_LABELS = 'emnist-byclass-test-labels-idx1-ubyte.gz'

    elif FLAGS.dataset == "emnist-bymerge":
        TRAIN_IMAGES = 'emnist-bymerge-train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'emnist-bymerge-train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 'emnist-bymerge-test-images-idx3-ubyte.gz'
        TEST_LABELS = 'emnist-bymerge-test-labels-idx1-ubyte.gz'

    elif FLAGS.dataset == "emnist-digits":
        TRAIN_IMAGES = 'emnist-digits-train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'emnist-digits-train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 'emnist-digits-test-images-idx3-ubyte.gz'
        TEST_LABELS = 'emnist-digits-test-labels-idx1-ubyte.gz'

    elif FLAGS.dataset == "emnist-letters":
        TRAIN_IMAGES = 'emnist-letters-train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'emnist-letters-train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 'emnist-letters-test-images-idx3-ubyte.gz'
        TEST_LABELS = 'emnist-letters-test-labels-idx1-ubyte.gz'

    elif FLAGS.dataset == "emnist-mnist":
        TRAIN_IMAGES = 'emnist-mnist-train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'emnist-mnist-train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 'emnist-mnist-test-images-idx3-ubyte.gz'
        TEST_LABELS = 'emnist-mnist-test-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                     source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                     source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                     source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir,
                                     source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                         .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
    return read_data_sets(train_dir)

# ****************************************************


def main(_):
    # Import data
    if FLAGS.dataset == "mnist":
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        print("FORMAT: " + str(type(mnist)))
        print("FORMAT: " + str(type(mnist[0])))
    elif FLAGS.dataset == "fashion-mnist":
        print("Reading fashion mnist")
        mnist = input_data.read_data_sets('data/fashion',
                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)
        print("Fashion-MNIST type: " + str(type(mnist)))
    elif FLAGS.dataset == "emnist":
        print("Reading e-mnist")
        mnist = read_data_sets_local('data/emnist', one_hot=True)
        print("e-MNIST type: " + str(type(mnist)))
    elif FLAGS.dataset == "emnist-digits":
        print("Reading e-mnist")
        mnist = read_data_sets_local('data/emnist', one_hot=True)
    elif FLAGS.dataset == "emnist-balanced":
        print("Reading e-mnist")
        mnist = read_data_sets_local('data/emnist', one_hot=True)
    elif FLAGS.dataset == "emnist-letters":
        print("Reading e-mnist")
        mnist = read_data_sets_local('data/emnist', one_hot=True)
    elif FLAGS.dataset == "emnist-bymerge":
        print("Reading e-mnist")
        mnist = read_data_sets_local('data/emnist', one_hot=True)
    elif FLAGS.dataset == "emnist-byclass":
        print("Reading e-mnist")
        mnist = read_data_sets_local('data/emnist', one_hot=True)
    tf.reset_default_graph()

    # Create the model
    caps_net = CapsNet(mnist, FLAGS.dataset)
    caps_net.creat_architecture()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_dir = cfg.TRAIN_DIR
    ckpt = tf.train.get_checkpoint_state(train_dir)

    with tf.Session(config=config) as sess:
        if ckpt and cfg.USE_CKPT:
            print("Reading parameters from %s" % ckpt.model_checkpoint_path)
            caps_net.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created model with fresh parameters.')
            sess.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements()
                                         for v in tf.trainable_variables()))
        # for test
        # caps_net.test(sess, 'validation')
        # exit()

        caps_net.train_writer.add_graph(sess.graph)
        iters = 0
        tic = time.time()
        for iters in xrange(cfg.MAX_ITERS):
            sys.stdout.write('>>> %d / %d \r' % (iters % cfg.PRINT_EVERY, cfg.PRINT_EVERY))
            sys.stdout.flush()
            caps_net.train_with_summary(sess, batch_size=100, iters=iters)
            if iters % cfg.PRINT_EVERY == 0 and iters > 0:
                toc = time.time()
                print('average time: %.2f secs' % (toc - tic))
                tic = time.time()
        caps_net.snapshot(sess, iters)
        caps_net.test(sess, 'test')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=cfg.DATA_DIR,
                        help='Directory for storing input data')
    parser.add_argument('--dataset', type = str, default = "mnist",
                        help='Dataset used to train the model')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.dataset)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    # for model building test
    # model_test()
