import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1


def net(image):
  conv1 = _conv_layer(image, 32, 9, 1)
  conv2 = _conv_layer(conv1, 64, 3, 2)
  conv3 = _conv_layer(conv2, 128, 3, 2)
  conv4 = _conv_layer(conv3, 256, 3, 2)
  # conv4 = _conv_layer(conv3, 256, 3, 2)
  resid1 = _residual_block(conv4, 3)
  resid2 = _residual_block(resid1, 3)
  resid3 = _residual_block(resid2, 3)
  resid4 = _residual_block(resid3, 3)
  resid5 = _residual_block(resid4, 3)
  # resid31 = _residual_block(resid30, 3)
  # resid32 = _residual_block(resid31, 3)
  # resid33 = _residual_block(resid32, 3)
  # resid34 = _residual_block(resid33, 3)
  # resid35 = _residual_block(resid34, 3)
  # resid36 = _residual_block(resid35, 3)
  # resid37 = _residual_block(resid36, 3)
  # resid38 = _residual_block(resid37, 3)
  # resid39 = _residual_block(resid38, 3)
  # resid40 = _residual_block(resid39, 3)
  # resid41 = _residual_block(resid40, 3)
  # resid42 = _residual_block(resid41, 3)
  # resid43 = _residual_block(resid42, 3)
  # resid44 = _residual_block(resid43, 3)
  # resid45 = _residual_block(resid44, 3)
  # resid46 = _residual_block(resid45, 3)
  # resid47 = _residual_block(resid46, 3)
  # resid48 = _residual_block(resid47, 3)
  # resid49 = _residual_block(resid48, 3)
  # resid50 = _residual_block(resid49, 3)
  # resid51 = _residual_block(resid50, 3)
  # resid52 = _residual_block(resid51, 3)
  # resid53 = _residual_block(resid52, 3)
  # resid54 = _residual_block(resid53, 3)
  # resid55 = _residual_block(resid54, 3)
  # resid56 = _residual_block(resid55, 3)
  # resid57 = _residual_block(resid56, 3)
  # resid58 = _residual_block(resid57, 3)
  # resid59 = _residual_block(resid58, 3)
  # resid60 = _residual_block(resid59, 3)
  # conv_t1 = _conv_tranpose_layer(resid5, 128, 3, 2)
  conv_t1 = _conv_tranpose_layer(resid5, 128, 3, 2)
  conv_t2 = _conv_tranpose_layer(conv_t1, 64, 3, 2)
  conv_t3 = _conv_tranpose_layer(conv_t2, 32, 3, 2)
  conv_t4 = _conv_layer(conv_t3, 3, 9, 1, relu=False)
  #preds = tf.nn.tanh(conv_t4) * 150 + 255. / 2
  preds = tf.nn.tanh(conv_t4) * 300
  return preds


def _conv_layer(net, num_filters, filter_size, strides, relu=True):
  weights_init = _conv_init_vars(net, num_filters, filter_size)
  strides_shape = [1, strides, strides, 1]
  net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
  net = _instance_norm(net)
  if relu:
    net = tf.nn.relu(net)
    #net = tf.nn.tanh(net)

  return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
  weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

  batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
  new_rows, new_cols = int(rows * strides), int(cols * strides)
  # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

  new_shape = [batch_size, new_rows, new_cols, num_filters]
  tf_shape = tf.stack(new_shape)
  strides_shape = [1, strides, strides, 1]

  net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
  net = _instance_norm(net)
  return tf.nn.relu(net)


def _residual_block(net, filter_size=3):
  tmp = _conv_layer(net, 256, filter_size, 1)
  return net + _conv_layer(tmp, 256, filter_size, 1, relu=False)


def _instance_norm(net, train=True):
  batch, rows, cols, channels = [i.value for i in net.get_shape()]
  var_shape = [channels]
  mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
  shift = tf.Variable(tf.zeros(var_shape))
  scale = tf.Variable(tf.ones(var_shape))
  epsilon = 1e-3
  normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
  return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
  _, rows, cols, in_channels = [i.value for i in net.get_shape()]
  if not transpose:
    weights_shape = [filter_size, filter_size, in_channels, out_channels]
  else:
    weights_shape = [filter_size, filter_size, out_channels, in_channels]

  weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
  return weights_init
