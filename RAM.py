from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tflearn

from GlimpseNet import GlimpseNet, LocNet
from RAMTrainModule import RAMTrainModule
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

_n_steps = 100000
_n_class = 10
_n_glimpse = 6
_cell_size = 256
_width = 28
_height = 28
_dim = 1
_loc_stddev = 0.22
loc_mean_arr = []
loc_arr = []
sampled_loc = []

def generate_reward_signal(predictions, y, n_glimpses, baselines, log_likelihood):
  r = tf.cast(tf.equal(predictions, y), tf.float32)
  rewards = tf.expand_dims(r, 1)
  rewards = tf.tile(rewards, (1, n_glimpses))
  advs = rewards - tf.stop_gradient(baselines)
  llratio = tf.reduce_mean(log_likelihood * advs)
  reward = tf.reduce_mean(r)
  mse = tf.reduce_mean(tf.square((rewards - baselines)))
  return reward, mse, llratio, advs

def step(output, i):
  loc, loc_mean = ln(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc.append(loc)
  return gl_next

def calcLogLikelihood(mean, samples, stdev):
  mean = tf.stack(mean)
  samples = tf.stack(samples)
  distribution = tf.contrib.distributions.Normal(mean, stdev)
  ll = distribution.log_prob(samples)
  ll = tf.transpose(tf.reduce_sum(ll, 2))
  return ll

X = tf.placeholder(tf.float32, shape=[None, _width * _height * _dim])
y = tf.placeholder(tf.int64, shape=[None])
n = tf.shape(X)[0]

with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(X, y)
with tf.variable_scope('loc_net'):
  ln = LocNet()

init_location = tf.random_uniform((n, 2), minval=-1, maxval=1)
init_glimpse = gl(init_location)

lstm = tf.contrib.rnn.LSTMCell(_cell_size, state_is_tuple=True)
state = lstm.zero_state(n, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (_n_glimpse))
outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, 
                                                   state, 
                                                   lstm, 
                                                   loop_function=step)

with tf.variable_scope('baseline'):
  w_b = tf.Variable(tf.truncated_normal((_cell_size, 1), stddev=0.01))
  b_b = tf.Variable(tf.constant(0.0, shape=(1,)))
baselines = []
for _, output in enumerate(outputs[1:]):
  b_t = tf.nn.xw_plus_b(output, w_b, b_b)
  b_t = tf.squeeze(b_t)
  baselines.append(b_t)
baselines = tf.transpose(tf.stack(baselines))

output = outputs[-1]

with tf.variable_scope('classifier'):
  w_logit = tf.Variable(tf.truncated_normal((_cell_size, _n_class), stddev=0.01))
  b_logit = tf.Variable(tf.constant(0.0, shape=(_n_class,)))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
crossentropy = tf.reduce_mean(crossentropy)
predictions = tf.argmax(logits, 1)
log_likelihood = calcLogLikelihood(loc_mean_arr, sampled_loc, _loc_stddev)

reward, mse, llratio, advs = generate_reward_signal(predictions=predictions, 
                                                    y=y, 
                                                    n_glimpses=_n_glimpse, 
                                                    baselines=baselines, 
                                                    log_likelihood=log_likelihood)

loss = -llratio + crossentropy + mse
var_list = tf.trainable_variables()
gradients = tf.gradients(loss, var_list)
gradients = tf.clip_by_global_norm(gradients, 5.)[0]

trainer = RAMTrainModule(X=X, y=y, glimpse_network=gl, loc_network=ln, 
                         t_variables=var_list, gradients=gradients)

with tf.Session() as sess:
  trainer.train(sess=sess, steps=_n_steps, advs=advs, mse=mse, ce=crossentropy,
                loss=loss, llratio=llratio, reward=reward, mnist=mnist)
