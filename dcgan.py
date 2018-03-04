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
batch_size = 50
batch_amount = 10000
learning_rate = 0.0002
lrelu_slope = 0.2
beta1_momentum = 0.5
inout_dimensions_x = 28
inout_dimensions_y = 28


def ceil(n):
    return math.ceil(n)


def discriminator(image):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
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
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        noise = tf.random_normal([batch_size, ceil(inout_dimensions_x/2**4) * ceil(inout_dimensions_y/2**4) * 1024], stddev=0.02)
        noise = tf.reshape(noise, (-1, ceil(inout_dimensions_x/2**4), ceil(inout_dimensions_y/2**4), 1024))

        stride = [1, 2, 2, 1]
        padding = "SAME"
        filter1 = tf.Variable(tf.random_normal([5, 5, 512, 1024]))
        output_shape1 = [batch_size, ceil(inout_dimensions_x/2**3), ceil(inout_dimensions_y/2**3), 512]
        filter2 = tf.Variable(tf.random_normal([5, 5, 256, 512]))
        output_shape2 = [batch_size, ceil(inout_dimensions_x/2**2), ceil(inout_dimensions_y/2**2), 256]
        filter3 = tf.Variable(tf.random_normal([5, 5, 128, 256]))
        output_shape3 = [batch_size, ceil(inout_dimensions_x/2), ceil(inout_dimensions_y/2), 128]
        filter4 = tf.Variable(tf.random_normal([5, 5, 1, 128]))
        output_shape4 = [batch_size, inout_dimensions_x, inout_dimensions_y, 1]

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


def train(discr_output_real, discr_output_fake, gen_output):
    discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_output_real, labels=tf.ones_like(discr_output_real)))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_output_fake, labels=tf.zeros_like(discr_output_fake)))
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output, labels=tf.ones_like(gen_output)))

    trainable = tf.trainable_variables()
    discriminator_vars = []
    generator_vars = []

    for t in trainable:
        if t.name.startswith('discriminator'):
            discriminator_vars.append(t)
        elif t.name.startswith('generator'):
            generator_vars.append(t)

    discriminator_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1_momentum).minimize(loss=discriminator_loss, var_list=discriminator_vars)
    generator_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1_momentum).minimize(loss=generator_loss, var_list=generator_vars)

    return discriminator_loss, generator_loss, discriminator_train, generator_train


real_image_input = tf.placeholder(tf.float32, [None, inout_dimensions_x * inout_dimensions_y], name='real_image_input')
gen_out = generator()
discr_out_real = discriminator(real_image_input)
discr_out_fake = discriminator(gen_out)
discr_loss, gen_loss, discr_train, gen_train = train(discr_out_real, discr_out_fake, gen_out)

tf.summary.scalar('Discriminator loss', discr_loss)
tf.summary.scalar('Generator loss', gen_loss)
merged = tf.summary.merge_all()
launch_date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.FileWriter("tensorboard/" + launch_date_time + "/")
output_path = "output/" + launch_date_time + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for batch in range(0, batch_amount):
        single_image = mnist.train.next_batch(batch_size)[0]
        summary, discr_loss_out, gen_loss_out, _, _ = sess.run([merged, discr_loss, gen_loss, discr_train, gen_train], feed_dict={real_image_input: single_image})
        print("Batch n: {} Discr loss: {} Gen loss: {}".format(batch, discr_loss_out, gen_loss_out))
        writer.add_summary(summary, batch)

        if batch % 100 == 0:
            generated_image = sess.run(gen_out)
            generated_image = generated_image[0].reshape([28, 28])
            plt.imsave(fname=output_path + str(batch) + ".png", arr=generated_image, cmap='Greys')

