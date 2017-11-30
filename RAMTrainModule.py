from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import sys

class RAMTrainModule(object):

  # Blob object
  # TODO: Clean this up
  def __init__(self, X, y, glimpse_network, loc_network, t_variables, gradients, 
               init_lr=0.01, min_lr=1.0e-5, decay=0.97, batch_size=128, optimizer="Adam"):

    self.X = X
    self.y = y
    self.gn = glimpse_network
    self.ln = loc_network
    self.t_variables = t_variables
    self.gradients = gradients
    self.init_lr = init_lr
    self.min_lr = min_lr
    self.decay = decay
    self.batch_size = batch_size
    self.steps_per_epoch = tf.shape(X)[0] / self.batch_size
    self.epoch = 0

    self.global_step = tf.get_variable('global_step', [], 
                                      initializer=tf.constant_initializer(0), 
                                      trainable=False)

    self.learning_rate = tf.train.exponential_decay(self.init_lr, 
                                                    self.global_step,
                                                    self.steps_per_epoch, 
                                                    self.decay,
                                                    staircase=True)

    self.learning_rate = tf.maximum(self.learning_rate, self.min_lr)
    self.opt = tf.train.AdamOptimizer(self.learning_rate)
    self.train_op = self.opt.apply_gradients(zip(self.gradients, 
                                                 self.t_variables), 
                                                 global_step=self.global_step)

  # Frankenstein Function
  # TODO: Use sessions to make this less horrible
  def train(self, sess, steps, advs, mse, ce, loss, llratio, reward, mnist,
            monte_carlo_sampling=10):

    sess.run(tf.global_variables_initializer())
    epoch = 1

    for i in xrange(steps):
      images, labels = mnist.train.next_batch(self.batch_size)
      images = np.tile(images, [monte_carlo_sampling, 1])
      labels = np.tile(labels, [monte_carlo_sampling])
      self.ln.sampling = True

      adv_val, mse_val, ce_val, ratio_val, \
      reward_val, loss_val, lr_val, _ = sess.run(
        [advs, mse, ce, llratio, reward, loss, self.learning_rate, self.train_op],
        feed_dict={self.X: images, self.y: labels})

      if i and i % 1 == 0:
        self.printiter(i, lr_val, ce_val, loss_val, mse_val, reward_val)

  
  def printiter(self, step, lr, crossentropy, loss, mse, reward):
    if (self.epoch == 0 or step % self.steps_per_epoch == 0):
      self.epoch += 1
      sys.stdout.write("-----------------------------------------------------------------------\n")
      sys.stdout.write("EPOCH: {:7}    LR: {:7.5f}   ---------------   ---------------------\n".format(self.epoch,lr))
    sys.stdout.write("\r LOSS: {:7.5f}   MSE: {:7.5f}   REWARD: {:7.5f}   CROSSENTROPY: {:7.5f}\r".format(loss,mse,reward,crossentropy))
    sys.stdout.flush()

