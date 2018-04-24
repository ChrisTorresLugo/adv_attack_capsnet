"""
Use fast gradient sign method to craft adversarial on MNIST.
Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

import argparse
from fast_gradient import fgm


img_size = 28
img_chan = 1
global n_classes
n_classes = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data/emnist",
                    help='Directory for storing input data')
parser.add_argument('--dataset', type=str, default="mnist",
                    help='Dataset used to train the model')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Batch size')
parser.add_argument('--train_epochs', type=int, default=5000,
                    help='Iterations')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='Epsilon - Amount of noise to be added')
parser.add_argument('--load', type=bool, default=True,
                    help='Load an old model instead of training a new one from scratch')
FLAGS, unparsed = parser.parse_known_args()
print(FLAGS.dataset)

if FLAGS.dataset == "mnist" or FLAGS.dataset == "emnist-mnist" or FLAGS.dataset == "emnist-digits":
    n_classes = 10
elif FLAGS.dataset == "emnist-balanced" or FLAGS.dataset == "emnist-bymerge":
    n_classes = 47
elif FLAGS.dataset == "emnist-byclass":
    n_classes = 62
elif FLAGS.dataset == "emnist-letters":
    n_classes = 27

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""

    if FLAGS.dataset == "emnist-mnist":
        labels_path = os.path.join(path,
                                   'emnist-mnist-%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   'emnist-mnist-%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 10

    elif FLAGS.dataset == "emnist-balanced":
        labels_path = os.path.join(path,
                                   'emnist-balanced-%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   'emnist-balanced-%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 47

    elif FLAGS.dataset == "emnist-byclass":
        labels_path = os.path.join(path,
                                   'emnist-byclass-%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   'emnist-byclass-%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 62

    elif FLAGS.dataset == "emnist-bymerge":
        labels_path = os.path.join(path,
                                   'emnist-bymerge-%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   'emnist-bymerge-%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 47


    elif FLAGS.dataset == "emnist-digits":
        labels_path = os.path.join(path,
                                   'emnist-digits-%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   'emnist-digits-%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 10

    elif FLAGS.dataset == "emnist-letters":
        labels_path = os.path.join(path,
                                   'emnist-letters-%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   'emnist-letters-%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 37

    elif FLAGS.dataset == "fashion-mnist" and kind == "train":
        labels_path = os.path.join("data/fashion",
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join("data/fashion",
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)
        n_classes = 10

    elif FLAGS.dataset == "fashion-mnist" and kind == "test":
        labels_path = os.path.join("data/fashion",
                                   't10k-labels-idx1-ubyte.gz')
        images_path = os.path.join("data/fashion",
                                   't10k-images-idx3-ubyte.gz')
        n_classes = 10


    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

print('\nLoading MNIST')

if FLAGS.dataset == "mnist":
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
else:
    (X_train, y_train) = load_mnist(path="data/emnist", kind="train")
    (X_test, y_test) = load_mnist(path="data/emnist", kind="test")

X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    if FLAGS.dataset == "mnist" or FLAGS.dataset == "emnist-mnist" or FLAGS.dataset == "emnist-digits"\
            or FLAGS.dataset == "fashion-mnist":
        logits_ = tf.layers.dense(z, units=10, name='logits')
    elif FLAGS.dataset == "emnist-balanced" or FLAGS.dataset == "emnist-bymerge":
        logits_ = tf.layers.dense(z, units=47, name='logits')
    elif FLAGS.dataset == "emnist-byclass":
        logits_ = tf.layers.dense(z, units=62, name='logits')
    elif FLAGS.dataset == "emnist-letters":
        logits_ = tf.layers.dense(z, units=27, name='logits')

    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')
    env.x_fgm = fgm(model, env.x, epochs=env.adv_epochs,
                      eps=env.adv_eps)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        print("Trying to load model")
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgm, feed_dict={
            env.x: X_data[start:end],
            env.adv_y: np.random.choice(n_classes),
            env.adv_eps: eps,
            env.adv_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, load=FLAGS.load, epochs=FLAGS.train_epochs, name='mnist')

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)

print('\nGenerating adversarial data')

X_adv = make_fgm(sess, env, X_test, eps=FLAGS.epsilon, epochs=8)

print('\nEvaluating on adversarial data')

evaluate(sess, env, X_adv, y_test)

print('\nRandomly sample adversarial data from each category')

y1 = predict(sess, env, X_test)
y2 = predict(sess, env, X_adv)

z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

X_tmp = np.empty((10, 28, 28))
y_tmp = np.empty((10, 10))
for i in range(10):
    print('Target {0}'.format(i))
    ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
    cur = np.random.choice(ind)
    X_tmp[i] = np.squeeze(X_adv[cur])
    y_tmp[i] = y2[cur]

print('\nPlotting results')

fig = plt.figure(figsize=(10, 1.2))
gs = gridspec.GridSpec(1, 10, wspace=0.05, hspace=0.05)

label = np.argmax(y_tmp, axis=1)
proba = np.max(y_tmp, axis=1)
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                  fontsize=12)

print('\nSaving figure')

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/fgm_%s2.png' % FLAGS.dataset)