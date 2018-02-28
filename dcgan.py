import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# Just to avoid warnings. Disable this from time to time during development
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST data set
mnist = input_data.read_data_sets("MNIST_data/")

# Hyperparams
mnist_batch_size = 1
mnist_batch_amount = 1
learning_rate = 0.0002
lrelu_slope = 0.2

image_placeholder = tf.placeholder(tf.float32, [None, 28 * 28], name='image_placeholder')


def discriminator(image):
    with tf.variable_scope("discriminator"):
        image = tf.reshape(image, [-1, 28, 28, 1])
        stride = [1, 2, 2, 1]
        padding = "SAME"
        filter1 = tf.Variable(tf.random_normal([5, 5, 1, 128]))
        filter2 = tf.Variable(tf.random_normal([5, 5, 128, 256]))
        filter3 = tf.Variable(tf.random_normal([5, 5, 256, 512]))
        filter4 = tf.Variable(tf.random_normal([5, 5, 512, 1024]))

        conv1 = tf.nn.conv2d(image, filter1, stride, padding)
        # Batch normalization is not applied to input layer since it
        # "resulted in sample oscillation and model instability" (from paper)
        relu1 = tf.nn.leaky_relu(conv1, lrelu_slope)

        print("Input shape: {}".format(image.shape))

        print("Shape after 1st convolution: {}".format(conv1.shape))

        conv2 = tf.nn.conv2d(relu1, filter2, stride, padding)
        batchnorm2 = tf.layers.batch_normalization(conv2, training=True)
        relu2 = tf.nn.leaky_relu(batchnorm2, lrelu_slope)

        print("Shape after 2nd convolution: {}".format(conv2.shape))

        conv3 = tf.nn.conv2d(relu2, filter3, stride, padding)
        batchnorm3 = tf.layers.batch_normalization(conv3, training=True)
        relu3 = tf.nn.leaky_relu(batchnorm3, lrelu_slope)

        print("Shape after 3rd convolution: {}".format(conv3.shape))

        conv4 = tf.nn.conv2d(relu3, filter4, stride, padding)
        batchnorm4 = tf.layers.batch_normalization(conv4, training=True)
        relu4 = tf.nn.leaky_relu(batchnorm4, lrelu_slope)

        print("Shape after 4th convolution: {}".format(conv4.shape))

        # For the discriminator, the last convolution layer is flattened (from paper)
        flattened = tf.reshape(relu4, (-1, 2 * 2 * 1024))

        # For the discriminator, the last convolution layer is flattened
        # and then fed into a single sigmoid output (from paper)
        single = tf.layers.dense(flattened, 1)
        out = tf.sigmoid(single)

        return flattened, single, out


def generator(noise):
    with tf.variable_scope("generator"):
        return


def train():
    return


runDiscriminator = discriminator(image_placeholder)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for batch in range(0, mnist_batch_amount):
        single_image = mnist.train.next_batch(mnist_batch_size)[0]
        flattened, single, out = sess.run(runDiscriminator, feed_dict={image_placeholder: single_image})
        print("Output of discriminator (sigmoid from single): {}".format(out))
