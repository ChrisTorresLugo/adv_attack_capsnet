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
import TFDB

FLAGS = None


def model_test():
    model = CapsNet(None)
    model.creat_architecture()
    print("pass")


def main(_):
    # Import data
    if FLAGS.dataset == "mnist":
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        print("FORMAT: " + str(type(mnist)))
        return
    elif FLAGS.dataset == "fashion-mnist":
        print("Reading fashion mnist")
        mnist = input_data.read_data_sets('data/fashion',
                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)
        print("Fashion-MNIST type: " + str(type(mnist)))
    elif FLAGS.dataset == "cifar-10":
        # train, test = tf.keras.datasets.cifar10.load_data()
        # mnist = tf.data.Dataset.from_tensor_slices((train[0], train[1]))
        mnist = TFDB.dataset.image.cifar10('tmp/cifar10')
        # dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        # print(features[0])
        # return

        # mnist = input_data.read_data_sets('http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz', one_hot=True)
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
            print('Created model with fresh paramters.')
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
