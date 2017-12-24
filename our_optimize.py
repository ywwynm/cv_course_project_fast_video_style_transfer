import tensorflow as tf
import network
import time
import numpy as np
import our_utils
from global_variable import logging as log
import residual_calculator as rec

def optimize(res_npy_ori_train_path, res_npy_trs_train_path, model_save_path_name,
             num_train_examples = 200, batch_size=4, learning_rate=1e-3):
  res_ori_train = np.load(res_npy_ori_train_path)
  res_trs_train = np.load(res_npy_trs_train_path)

  batch_shape = (batch_size, 208, 312, 3)
  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = network.net(X_content / 255.0)

    # reshape
    Y_content_flat = tf.reshape(Y_content, [-1, 312 * 208 * 3])
    preds_flat = tf.reshape(preds, [-1, 312 * 208 * 3])
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
      log.info("iterations:" + str(iterations) + ", loss: " + str(np.sqrt(tup[0] / 312 / 208 / 3)) +
            ", start_time: " + str(start_time) +
            ", end_time: " + str(end_time) + ", delta_time: " + str(delta_time))

    all_stop_time = time.time()
    log.info('optimizing stopped at ' + str(all_stop_time)
             + ', cost ' + str((all_stop_time - all_start_time) / 60) + ' minutes')
    tf.train.Saver().save(sess, model_save_path_name)
    del res_ori_train, res_trs_train


def evaluate_model(res_npy_ori_test_path, res_npy_trs_test_path, model_path_name):
  batch_shape = (1, 208, 312, 3)
  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = network.net(X_content / 255.0)

    Y_content_flat = tf.reshape(Y_content, [-1, 312 * 208 * 3])
    preds_flat = tf.reshape(preds, [-1, 312 * 208 * 3])
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
      log.info("test " + str(i) + "average loss: " + str(np.sqrt(tup[0] / 312 / 208 / 3)))

def generate_frames(first_frame_path_name, res_npy_ori_test_path, res_npy_trs_test_path, model_path_name):
  batch_shape = (1, 208, 312, 3)
  with tf.Session() as sess:
    X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
    Y_content = tf.placeholder(tf.float32, shape=batch_shape, name="Y_content")
    preds = network.net(X_content / 255.0)

    im_array = rec.img_to_tensor(first_frame_path_name)
    Recon = im_array
    Recon_array=[]

    Y_content_flat = tf.reshape(Y_content, [-1, 312 * 208 * 3])
    preds_flat = tf.reshape(preds, [-1, 312 * 208 * 3])
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
      log.info("test " + str(i) + "average loss: " + str(np.sqrt(tup[0] / 312 / 208 / 3)))
      Recon = im_array + tup[1]
      Recon_array.append(Recon)

