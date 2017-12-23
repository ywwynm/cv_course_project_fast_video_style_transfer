import residual_calculator as res_calc
import utils
import tensorflow as tf
import numpy as np

data_dir = 'input\procedure_208p_2min_wave\data'
ori_dir = 'input\procedure_208p_2min_wave\in'
trs_dir = 'input\procedure_208p_2min_wave\out'

ori_res_train = np.load(data_dir + '\ori_res_train.npy')
trs_res_train = np.load(data_dir + '\\trs_res_train.npy')

ori_res_test = np.load(data_dir + '\ori_res_test.npy')
trs_res_test = np.load(data_dir + '\\trs_res_test.npy')


# Now we want to learn the reflection from ori_res_train to trs_res_train
# And then we can apply the learned knowledge to new ori_res and get new trs_res


def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
  correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
  return sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float16)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float16)
  return tf.Variable(initial)


def conv2d(x, W):
  # stride [1, x_move, y_move, 1]
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # SAME: 卷积后'图片'大小保持一致


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_6x6(x):
  return tf.nn.max_pool(x, ksize=[1, 6, 6, 1], strides=[1, 2, 2, 1], padding='SAME')


xs = tf.placeholder(tf.float16, [None, 720 * 480])
ys = tf.placeholder(tf.float16, [None, 720 * 480])
keep_prob = tf.placeholder(tf.float16)
x_residual = tf.reshape(xs, [-1, 720, 480, 3])  # [batch_size, width, height, channels]

# conv1 layer
W_conv1 = weight_variable([5, 5, 3, 16])  # patch: 5 * 5, 3: channels(RGB), 16: output feature size
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_residual, W_conv1) + b_conv1)  # output size 720*480*16
h_pool1 = max_pool_2x2(h_conv1)  # output size 360*240*16

# conv2 layer
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 360*240*32
h_pool2 = max_pool_4x4(h_conv2)  # output size 90*60*32

# conv3 layer
W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # output size 90*60*64
h_pool3 = max_pool_6x6(h_conv3)  # output size 15*10*64

# func1 layer
W_fc1 = weight_variable([90 * 60 * 64, 720 * 480])
b_fc1 = bias_variable([720 * 480])
h_pool2_flat = tf.reshape(h_pool2, [-1, 90 * 60 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

ys_flat = tf.reshape(ys, [-1, 720 * 480])
mse = tf.reduce_mean(tf.reduce_sum(tf.square(h_fc1_drop - ys_flat)))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10):
    batch_xs, batch_ys = utils.new_batch(ori_res_test, trs_res_test, 50)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
      print(compute_accuracy(ori_res_test, trs_res_test))
