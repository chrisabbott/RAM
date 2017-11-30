from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class GlimpseNet(object):

  def __init__(self, X, y):
    self._original_size = 28
    self._num_channels = 1
    self._win_size = 8
    self._depth = 1
    self._sensor_size = self._win_size**2 * self._depth
    self._hg_size = 128
    self._g_size = 256
    self._hl_size = 128
    self._loc_dim = 2
    self._loc_std = 0.22
    self.X = X
    self.y = y

    self.init_weights()

  def init_weights(self):
    self.w_g0 = tf.Variable(tf.truncated_normal((self._sensor_size, self._hg_size), stddev=0.01))
    self.b_g0 = tf.Variable(tf.constant(0.0, shape=(self._hg_size,)))
    self.w_l0 = tf.Variable(tf.truncated_normal((self._loc_dim, self._hl_size), stddev=0.01))
    self.b_l0 = tf.Variable(tf.constant(0.0, shape=(self._hl_size,)))
    self.w_g1 = tf.Variable(tf.truncated_normal((self._hg_size, self._g_size), stddev=0.01))
    self.b_g1 = tf.Variable(tf.constant(0.0, shape=(self._g_size,)))
    self.w_l1 = tf.Variable(tf.truncated_normal((self._hl_size, self._g_size), stddev=0.01))
    self.b_l1 = tf.Variable(tf.constant(0.0, shape=(self._g_size,)))

  def get_glimpse(self, loc):
    images = tf.reshape(self.X, [tf.shape(self.X)[0], 
                                self._original_size, self._original_size,
                                self._num_channels])

    glimpses = tf.image.extract_glimpse(images, [self._win_size, self._win_size], loc)
    glimpses = tf.reshape(glimpses, [tf.shape(loc)[0], 
                                    self._win_size * self._win_size * self._num_channels])
    return glimpses

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)
    glimpse_input = tf.reshape(glimpse_input,
                              (tf.shape(loc)[0], self._sensor_size))
    g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
    g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g

class LocNet(object):

  def __init__(self):
    self._loc_dim = 2
    self._input_dim = 256
    self._loc_std = 0.22
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = tf.Variable(tf.truncated_normal((self._input_dim, self._loc_dim), stddev=0.01))
    self.b = tf.Variable(tf.constant(0.0, shape=(self._loc_dim,)))

  def __call__(self, input):
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)

    if self.sampling:
      lt = mean + tf.random_normal(
        (tf.shape(input)[0], self._loc_dim), stddev=self._loc_std)
      lt = tf.clip_by_value(lt, -1., 1.)
    else:
      lt = mean
    lt = tf.stop_gradient(lt)
    return lt, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
