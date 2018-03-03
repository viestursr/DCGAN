import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import math
import datetime

# Just to avoid warnings. Disable this from time to time during development
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST data set
mnist = input_data.read_data_sets("MNIST_data/")

# Hyperparams
mnist_batch_size = 50
mnist_batch_amount = 10000
learning_rate = 0.00002
lrelu_slope = 0.2
beta1_momentum = 0.5
noise_batch_size = 1
inout_dimensions_x = 28
inout_dimensions_y = 28


def ceil(n):
    return math.ceil(n)


def discriminator(image):
    with tf.variable_scope("discriminator"):
        image = tf.reshape(image, [-1, inout_dimensions_x, inout_dimensions_y, 1])
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
        flattened = tf.reshape(relu4, (-1, ceil(inout_dimensions_x/2**4) * ceil(inout_dimensions_y/2**4) * 1024))

        # For the discriminator, the last convolution layer is flattened
        # and then fed into a single sigmoid output (from paper)
        single = tf.layers.dense(flattened, 1)
        out = tf.sigmoid(single)

        return out


def generator():
    with tf.variable_scope("generator"):
        noise = tf.random_normal([noise_batch_size, ceil(inout_dimensions_x/2**4) * ceil(inout_dimensions_y/2**4) * 1024], stddev=0.02)
        noise = tf.reshape(noise, (-1, ceil(inout_dimensions_x/2**4), ceil(inout_dimensions_y/2**4), 1024))

        stride = [1, 2, 2, 1]
        padding = "SAME"
        filter1 = tf.Variable(tf.random_normal([5, 5, 512, 1024]))
        output_shape1 = [noise_batch_size, ceil(inout_dimensions_x/2**3), ceil(inout_dimensions_y/2**3), 512]
        filter2 = tf.Variable(tf.random_normal([5, 5, 256, 512]))
        output_shape2 = [noise_batch_size, ceil(inout_dimensions_x/2**2), ceil(inout_dimensions_y/2**2), 256]
        filter3 = tf.Variable(tf.random_normal([5, 5, 128, 256]))
        output_shape3 = [noise_batch_size, ceil(inout_dimensions_x/2), ceil(inout_dimensions_y/2), 128]
        filter4 = tf.Variable(tf.random_normal([5, 5, 1, 128]))
        output_shape4 = [noise_batch_size, inout_dimensions_x, inout_dimensions_y, 1]

        conv1 = tf.nn.conv2d_transpose(noise, filter1, output_shape1, stride, padding)
        batchnorm1 = tf.layers.batch_normalization(conv1, training=True)
        relu1 = tf.nn.relu(batchnorm1)

        print("Input noise shape: {}".format(noise.shape))

        print("Noise shape after 1st convolution: {}".format(conv1.shape))

        conv2 = tf.nn.conv2d_transpose(relu1, filter2, output_shape2, stride, padding)
        batchnorm2 = tf.layers.batch_normalization(conv2, training=True)
        relu2 = tf.nn.relu(batchnorm2)

        print("Noise shape after 2nd convolution: {}".format(conv2.shape))

        conv3 = tf.nn.conv2d_transpose(relu2, filter3, output_shape3, stride, padding)
        batchnorm3 = tf.layers.batch_normalization(conv3, training=True)
        relu3 = tf.nn.relu(batchnorm3)

        print("Noise shape after 3rd convolution: {}".format(conv3.shape))

        conv4 = tf.nn.conv2d_transpose(relu3, filter4, output_shape4, stride, padding)
        # Batch normalization is not applied to output layer since it
        # "resulted in sample oscillation and model instability" (from paper)
        tanh = tf.nn.tanh(conv4)

        print("Noise shape after 4th convolution: {}".format(conv4.shape))

        return tanh


def train(discr_output_real):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_output_real, labels=tf.ones_like(discr_output_real))
    discriminator_loss = tf.reduce_mean(ce)

    trainable = tf.trainable_variables()
    discriminator_vars = []

    for t in trainable:
        if t.name.startswith('discriminator'):
            discriminator_vars.append(t)

    discriminator_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1_momentum).minimize(loss=discriminator_loss, var_list=discriminator_vars)

    return discriminator_loss, discriminator_train, ce


real_image_input = tf.placeholder(tf.float32, [None, inout_dimensions_x * inout_dimensions_y], name='real_image_input')
discr_out = discriminator(real_image_input)
discr_loss, discr_train, ce = train(discr_out)

tf.summary.scalar('Discriminator loss', discr_loss)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for batch in range(0, mnist_batch_amount):
        single_image = mnist.train.next_batch(mnist_batch_size)[0]
        summary, loss, _ = sess.run([merged, discr_loss, discr_train], feed_dict={real_image_input: single_image})
        print("Batch n: {} Loss: {}".format(batch, loss))
        writer.add_summary(summary, batch)

