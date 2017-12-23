# from __future__ import print_function
# import functools
# import vgg, pdb, time
# import tensorflow as tf, numpy as np, os
import tensorflow as tf
import transform
import time
import random

# import data
# import residual_calculator as res_calc
import numpy as np


def new_batch(data1, data2, batch_size):
  idx_arr = random.sample(range(0, len(data1)), batch_size)
  ret_arr_1 = []
  ret_arr_2 = []
  for i in range(len(idx_arr)):
    ret_arr_1.append(data1[idx_arr[i]])
    ret_arr_2.append(data2[idx_arr[i]])
  return ret_arr_1, ret_arr_2


# from utils import get_img
#
# STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
# CONTENT_LAYER = 'relu4_2'
# DEVICES = 'CUDA_VISIBLE_DEVICES'
#
# # np arr, np arr
def optimize(epochs=2, print_iterations=1000,
             batch_size=4,
             learning_rate=1e-3, debug=False):  # save_path='saver/fns.ckpt', slow=False


  data_dir = 'D:\projects\python\cv_course_project_fast_video_style_transfer\input\procedure_208p_2min_wave\data'

  ori_res_train = np.load(data_dir + '\ori_res_train.npy')
  trs_res_train = np.load(data_dir + '\\trs_res_train.npy')

  batch_shape = (batch_size, 208, 312, 3)  # batch?
  # with tf.Session() as sess:
  #
  #   X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
  #   Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
  #   preds = transform.net(X_content / 255.0)
  #
  #   # reshape
  #   Y_content_flat = tf.reshape(Y_content, [-1, 312 * 208 * 3])
  #   preds_flat = tf.reshape(preds, [-1, 312 * 208 * 3])
  #   loss = tf.reduce_mean(tf.reduce_sum(tf.square(preds_flat - Y_content_flat), reduction_indices=[1]))
  #
  #   train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  #   sess.run(tf.global_variables_initializer())
  #   import random
  #   uid = random.randint(1, 100)
  #   #         print("UID: %s" % uid)
  #   for epoch in range(epochs):
  #     num_examples = 100  # len(content_targets)
  #     iterations = 0
  #     while iterations * batch_size < num_examples:
  #       start_time = time.time()
  #       # curr = iterations * batch_size
  #       # step = curr + batch_size
  #       X_batch_array, Y_batch_array = new_batch(ori_res_train, trs_res_train, batch_size)
  #       X_batch = np.zeros(batch_shape, dtype=np.float32)
  #       Y_batch = np.zeros(batch_shape, dtype=np.float32)
  #       for i in range(0, batch_size):
  #         X_batch[i] = X_batch_array[i]
  #         Y_batch[i] = Y_batch_array[i]
  #
  #       iterations += 1
  #
  #       assert X_batch.shape[0] == batch_size
  #
  #       feed_dict = {
  #         X_content: X_batch, Y_content: Y_batch
  #       }
  #
  #       train_step.run(feed_dict=feed_dict)
  #       end_time = time.time()
  #       delta_time = end_time - start_time
  #       if debug:
  #         print("UID: %s, batch time: %s" % (uid, delta_time))
  #       is_print_iter = int(iterations) % print_iterations == 0
  #       # if slow:
  #       #     is_print_iter = epoch % print_iterations == 0
  #       is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
  #       should_print = True  # is_print_iter or is_last
  #       if should_print:
  #         to_get = [loss, preds]
  #         test_feed_dict = {
  #           X_content: X_batch, Y_content: Y_batch
  #         }
  #
  #         tup = sess.run(to_get, feed_dict=test_feed_dict)
  #         print("iterations:", iterations, "loss:", np.sqrt(tup[0] / 312 / 208 / 3), "start_time: ", start_time, "end_time: ", end_time, "delta_time: ", delta_time)
  #
  #         # _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
  #         # losses = (_style_loss, _content_loss, _tv_loss, _loss)
  #         # if slow:
  #         #    _preds = vgg.unprocess(_preds)
  #         # else:
  #         #    saver = tf.train.Saver()
  #         #    res = saver.save(sess, save_path)
  #         # yield(_preds, losses, iterations, epoch)
  #   tf.train.Saver().save(sess, 'model\model_1.ckpt')

  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = transform.net(X_content / 255.0)

    Y_content_flat = tf.reshape(Y_content, [-1, 312 * 208 * 3])
    preds_flat = tf.reshape(preds, [-1, 312 * 208 * 3])
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(preds_flat - Y_content_flat), reduction_indices=[1]))

    del ori_res_train, trs_res_train

    tf.train.Saver().restore(sess, 'model/model_1.ckpt')

    ori_res_test = np.load(data_dir + '\ori_res_test.npy')
    trs_res_test = np.load(data_dir + '\\trs_res_test.npy')
    X_batch = np.zeros(batch_shape, dtype=np.float32)
    Y_batch = np.zeros(batch_shape, dtype=np.float32)
    for i in range(0, int(len(ori_res_test) / batch_size)):
      X_batch_array, Y_batch_array = new_batch(ori_res_test, trs_res_test, batch_size)
      for i in range(0, batch_size):
        X_batch[i] = X_batch_array[i]
        Y_batch[i] = Y_batch_array[i]

      to_get = [loss, preds]
      test_feed_dict = {
         X_content: X_batch, Y_content: Y_batch
      }
      tup = sess.run(to_get, feed_dict=test_feed_dict)

      print("average loss:", np.sqrt(tup[0] / 312 / 208 / 3))

#
# def _tensor_size(tensor):
#     from operator import mul
#     return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
