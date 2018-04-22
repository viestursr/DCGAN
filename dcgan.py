import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import math
import datetime
import numpy as np
import time
import glob

# Just to avoid warnings. Disable this from time to time during development
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST data set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

data_set = "celeba"

# Hyperparams
batch_size = 128
noise_dimension = 100
epoch_count = 100
learning_rate = 0.0002
lrelu_slope = 0.2
beta1_momentum = 0.5
normal_distribution_stddev = 0.02
label_smoothing = 0.9
inout_dimensions_x = 64
inout_dimensions_y = 64

channels = 3

if data_set == "mnist":
    channels = 1


def get_dataset():
    if data_set == "mnist":
        print("Preparing mnist dataset...")
        dataset = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
        dataset = (dataset - 0.5) / 0.5
        print(dataset.shape)
        print("mnist dataset ready")
        return dataset

    if data_set == "celeba":
        print("Preparing celebA dataset. This will take a while...")
        files = list(glob.glob("celeba/*"))

        dataset = np.array([((plt.imread(f) - 0.5) / 0.5) for f in files])
        np.random.shuffle(dataset)

        print(dataset.shape)

        print("celebA dataset ready")
        return dataset


def ceil(n):
    return math.ceil(n)


def discriminator(image, reuse_in=None):
    with tf.variable_scope("discriminator", reuse=reuse_in):
        stride = (2, 2)
        kernel_size = [5, 5]
        padding = "SAME"
        filter1 = 128
        filter2 = 256
        filter3 = 512
        filter4 = 1024

        conv1 = tf.layers.conv2d(image, filter1, kernel_size, stride, padding)
        # Batch normalization is not applied to input layer since it
        # "resulted in sample oscillation and model instability" (from paper)
        relu1 = tf.nn.leaky_relu(conv1, lrelu_slope)

        print("Input shape: {}".format(image.shape))

        print("Shape after 1st convolution: {}".format(conv1.shape))

        conv2 = tf.layers.conv2d(relu1, filter2, kernel_size, stride, padding)
        batchnorm2 = tf.layers.batch_normalization(conv2, training=True)
        relu2 = tf.nn.leaky_relu(batchnorm2, lrelu_slope)

        print("Shape after 2nd convolution: {}".format(conv2.shape))

        conv3 = tf.layers.conv2d(relu2, filter3, kernel_size, stride, padding)
        batchnorm3 = tf.layers.batch_normalization(conv3, training=True)
        relu3 = tf.nn.leaky_relu(batchnorm3, lrelu_slope)

        print("Shape after 3rd convolution: {}".format(conv3.shape))

        conv4 = tf.layers.conv2d(relu3, filter4, kernel_size, stride, padding)
        batchnorm4 = tf.layers.batch_normalization(conv4, training=True)
        relu4 = tf.nn.leaky_relu(batchnorm4, lrelu_slope)

        print("Shape after 4th convolution: {}".format(conv4.shape))

        # For the discriminator, the last convolution layer is flattened (from paper)
        flattened = tf.reshape(relu4, (batch_size, ceil(inout_dimensions_x/2**4) * ceil(inout_dimensions_y/2**4) * 1024))

        # For the discriminator, the last convolution layer is flattened
        # and then fed into a single sigmoid output (from paper)
        single = tf.layers.dense(flattened, 1)
        out = tf.sigmoid(single)

        return out, single


def generator():
    with tf.variable_scope("generator"):
        dense_noise = tf.layers.dense(inputs=noise_input, units=ceil(inout_dimensions_x/2**4) * ceil(inout_dimensions_y/2**4) * 1024)
        noise_reshaped = tf.reshape(dense_noise, (batch_size, ceil(inout_dimensions_x/2**4), ceil(inout_dimensions_y/2**4), 1024))

        stride = (2, 2)
        kernel_size = [5, 5]
        padding = "SAME"
        filter1 = 512
        filter2 = 256
        filter3 = 128
        filter4 = channels # This is the channel amount of final layer

        conv1 = tf.layers.conv2d_transpose(noise_reshaped, filter1, kernel_size, stride, padding)
        batchnorm1 = tf.layers.batch_normalization(conv1, training=True)
        relu1 = tf.nn.relu(batchnorm1)

        print("Input noise shape: {}".format(noise_reshaped.shape))

        print("Noise shape after 1st convolution: {}".format(conv1.shape))

        conv2 = tf.layers.conv2d_transpose(relu1, filter2, kernel_size, stride, padding)
        batchnorm2 = tf.layers.batch_normalization(conv2, training=True)
        relu2 = tf.nn.relu(batchnorm2)

        print("Noise shape after 2nd convolution: {}".format(conv2.shape))

        conv3 = tf.layers.conv2d_transpose(relu2, filter3, kernel_size, stride, padding)
        batchnorm3 = tf.layers.batch_normalization(conv3, training=True)
        relu3 = tf.nn.relu(batchnorm3)

        print("Noise shape after 3rd convolution: {}".format(conv3.shape))

        conv4 = tf.layers.conv2d_transpose(relu3, filter4, kernel_size, stride, padding)
        # Batch normalization is not applied to output layer since it
        # "resulted in sample oscillation and model instability" (from paper)
        tanh = tf.nn.tanh(conv4)

        print("Noise shape after 4th convolution: {}".format(conv4.shape))

        return tanh


def train_discr(discr_real, discr_fake, discr_real_logits, discr_fake_logits):
    discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_real_logits, labels=tf.ones_like(discr_real) * label_smoothing))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_fake_logits, labels=tf.zeros_like(discr_fake)))
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake

    trainable = tf.trainable_variables()
    discriminator_vars = []

    for t in trainable:
        if t.name.startswith('discriminator'):
            discriminator_vars.append(t)

    discriminator_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1_momentum).minimize(loss=discriminator_loss, var_list=discriminator_vars)
    return discriminator_loss, discriminator_train


def train_gen(discr_fake, discr_fake_logits):
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discr_fake_logits, labels=tf.ones_like(discr_fake) * label_smoothing))

    trainable = tf.trainable_variables()
    generator_vars = []

    for t in trainable:
        if t.name.startswith('generator'):
            generator_vars.append(t)

    generator_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1_momentum).minimize(loss=generator_loss, var_list=generator_vars)

    return generator_loss, generator_train


real_image_input = tf.placeholder(tf.float32, [None, inout_dimensions_x, inout_dimensions_y, channels], name='real_image_input')
noise_input = tf.placeholder(tf.float32, [None, 1, 1, noise_dimension], name='noise_input')
gen_out = generator()
discr_out_real, discr_out_real_logits = discriminator(real_image_input)
discr_out_fake, discr_out_fake_logits = discriminator(gen_out, reuse_in=True)

discr_loss, discr_train = train_discr(discr_out_real, discr_out_fake, discr_out_real_logits, discr_out_fake_logits)
gen_loss, gen_train = train_gen(discr_out_fake, discr_out_fake_logits)

tf.summary.scalar('Discriminator loss', discr_loss)
tf.summary.scalar('Generator loss', gen_loss)
tf.summary.image('Generated image', gen_out, max_outputs=10)
merged = tf.summary.merge_all()
launch_date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.FileWriter("tensorboard/" + launch_date_time + "/")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    bench_noise = np.random.normal(0, 1, size=[batch_size, 1, 1, noise_dimension])

    training_set = get_dataset()

    global_counter = 0

    print("Training started")

    summary = sess.run(merged, feed_dict={real_image_input: training_set[0*batch_size:(0+1)*batch_size], noise_input: bench_noise})
    writer.add_summary(summary, global_counter)

    for epoch in range(1, epoch_count):
        epoch_start_time = time.time()

        for i in range(len(training_set) // batch_size):
            global_counter = global_counter + batch_size

            image_batch = training_set[i*batch_size:(i+1)*batch_size]

            # Update discriminator
            noise = np.random.normal(0, 1, size=[batch_size, 1, 1, noise_dimension])
            discr_loss_out, _ = sess.run([discr_loss, discr_train], feed_dict={real_image_input: image_batch, noise_input: noise})

            # Update generator
            noise = np.random.normal(0, 1, size=[batch_size, 1, 1, noise_dimension])
            gen_loss_out, _ = sess.run([gen_loss, gen_train], feed_dict={real_image_input: image_batch, noise_input: noise})

            if i == (len(training_set) // batch_size) - 1:
                print("Epoch n: {} | Discr loss: {} | Gen loss: {} | Images shown: {} | Epoch time: {} sec".format(epoch, discr_loss_out, gen_loss_out, global_counter ,time.time() - epoch_start_time))
                summary = sess.run(merged, feed_dict={real_image_input: image_batch, noise_input: bench_noise})
                writer.add_summary(summary, global_counter)

