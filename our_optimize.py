import tensorflow as tf
import network
import time, os
import numpy as np
import our_utils
from global_variable import logging as log
import residual_calculator as rec
import scipy.misc

# frame_ori_path = 'frames/wave_208p_10min_wave/in'
# frame_trs_path = 'frames/wave_208p_10min_wave/out'

img_width = 312
img_height = 208
# img_width = 720
# img_height = 480

def optimize(res_npy_ori_train_path, res_npy_trs_train_path, model_save_path_name,
             # frame_ori_path, frame_trs_path, model_save_path_name,
             num_train_examples = 200, batch_size=4, learning_rate=1e-3):
  res_ori_train = np.load(res_npy_ori_train_path)
  res_trs_train = np.load(res_npy_trs_train_path)
  # res_ori_train = rec.get_frames_tensors(frame_ori_path, 1, 15001)
  # res_trs_train = rec.get_frames_tensors(frame_trs_path, 1, 15001)

  batch_shape = (batch_size, img_height, img_width, 3)
  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = network.net((X_content + 255.0) / 255.0 / 2)

    # reshape
    Y_content_flat = tf.reshape(Y_content, [-1, img_width * img_height * 3])
    preds_flat = tf.reshape(preds, [-1, img_width * img_height * 3])
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(preds_flat - Y_content_flat), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess.run(tf.global_variables_initializer())
    iterations = 0

    all_start_time = time.time()
    log.info('start optimizing at ' + str(all_start_time))
    while iterations * batch_size < num_train_examples:
      start_time = time.time()
      X_batch_array, Y_batch_array = our_utils.new_batch(res_ori_train, res_trs_train, batch_size)
      X_batch = np.zeros(batch_shape, dtype=np.float32)
      Y_batch = np.zeros(batch_shape, dtype=np.float32)
      for i in range(0, batch_size):
        X_batch[i] = X_batch_array[i]
        Y_batch[i] = Y_batch_array[i]

      iterations += 1

      assert X_batch.shape[0] == batch_size

      feed_dict = {
        X_content: X_batch, Y_content: Y_batch
      }

      train_step.run(feed_dict=feed_dict)
      end_time = time.time()
      delta_time = end_time - start_time
      to_get = [loss, preds]
      test_feed_dict = {
        X_content: X_batch, Y_content: Y_batch
      }

      tup = sess.run(to_get, feed_dict=test_feed_dict)
      log.info("iterations:" + str(iterations) + ", loss: " + str(np.sqrt(tup[0] / img_width / img_height / 3)) +
            ", start_time: " + str(start_time) +
            ", end_time: " + str(end_time) + ", delta_time: " + str(delta_time))

    all_stop_time = time.time()
    log.info('optimizing stopped at ' + str(all_stop_time)
             + ', cost ' + str((all_stop_time - all_start_time) / 60) + ' minutes')
    tf.train.Saver().save(sess, model_save_path_name)
    del res_ori_train, res_trs_train


def evaluate_model(res_npy_ori_test_path, res_npy_trs_test_path, model_path_name):
  batch_shape = (1, img_height, img_width, 3)
  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = network.net(X_content / 255.0)

    Y_content_flat = tf.reshape(Y_content, [-1, img_width * img_height * 3])
    preds_flat = tf.reshape(preds, [-1, img_width * img_height * 3])
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(preds_flat - Y_content_flat), reduction_indices=[1]))

    tf.train.Saver().restore(sess, model_path_name)

    res_ori_test = np.load(res_npy_ori_test_path)
    res_trs_test = np.load(res_npy_trs_test_path)

    X_batch = np.zeros(batch_shape, dtype=np.float32)
    Y_batch = np.zeros(batch_shape, dtype=np.float32)
    for i in range(0, int(len(res_ori_test))):
      X_batch_array, Y_batch_array = our_utils.next_batch(res_ori_test, res_trs_test, i)
      X_batch[0] = X_batch_array[0]
      Y_batch[0] = Y_batch_array[0]

      to_get = [loss, preds]
      test_feed_dict = {
         X_content: X_batch, Y_content: Y_batch
      }
      tup = sess.run(to_get, feed_dict=test_feed_dict)
      log.info("test " + str(i) + "average loss: " + str(np.sqrt(tup[0] / img_width / img_height / 3)))


def generate_frames(first_frame_path_name, res_npy_ori_test_path, res_npy_trs_test_path, model_path_name):
  batch_shape = (1, img_height, img_width, 3)
  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = network.net(X_content / 255.0)

    generated_frames_array=[]

    Y_content_flat = tf.reshape(Y_content, [-1, img_width * img_height * 3])
    preds_flat = tf.reshape(preds, [-1, img_width * img_height * 3])
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(preds_flat - Y_content_flat), reduction_indices=[1]))

    tf.train.Saver().restore(sess, model_path_name)

    # res_ori_test = np.load(res_npy_ori_test_path)
    # res_trs_test = np.load(res_npy_trs_test_path)
    res_ori_test = rec.get_frames_tensors(frame_ori_path, 15001, 17982)
    res_trs_test = rec.get_frames_tensors(frame_trs_path, 15001, 17982)

    X_batch = np.zeros(batch_shape, dtype=np.float32)
    Y_batch = np.zeros(batch_shape, dtype=np.float32)

    for i in range(int(len(res_ori_test))):
      X_batch_array, Y_batch_array = our_utils.next_batch(res_ori_test, res_trs_test, i)
      X_batch[0] = X_batch_array[0]
      Y_batch[0] = Y_batch_array[0]

      to_get = [loss, preds]
      test_feed_dict = {
         X_content: X_batch, Y_content: Y_batch
      }
      tup = sess.run(to_get, feed_dict=test_feed_dict)
      log.info("test " + str(i) + ", average loss: " + str(np.sqrt(tup[0] / img_width / img_height / 3)))
      generated_frames_array.append(tup[1])

    output_dir = 'output/frames/wave_208p_10min_wave_iter30000/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for i in range(len(generated_frames_array)):
      scipy.misc.imsave(output_dir + 'frame_' + str(i) + '.jpg', generated_frames_array[i][0])


