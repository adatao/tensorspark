import tensorflow as tf
import numpy as numpy
from parameterservermodel import ParameterServerModel

def weight_variable(shape, name):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

class MnistCNN(ParameterServerModel):

   def __init__(self):
      session = tf.InteractiveSession()
      x = tf.placeholder("float", shape=[None, 784], name='x')
      x_image = tf.reshape(x, [-1,28,28,1], name='reshape')
      y_ = tf.placeholder("float", shape=[None, 10], name='y_')
      W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
      b_conv1 = bias_variable([32], 'b_conv1')
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
      W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
      b_conv2 = bias_variable([64], 'b_conv2')
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
      b_fc1 = bias_variable([1024], 'b_fc1')
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
      W_fc2 = weight_variable([1024, 10], 'W_fc2')
      b_fc2 = bias_variable([10], 'b_fc2')

      # not using dropout for testing, only training
      y_conv_dropout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

      variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
      loss = tf.nn.softmax_cross_entropy_with_logits(y_conv_dropout, y_)

      optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
      compute_gradients = optimizer.compute_gradients(loss, variables)
      apply_gradients = optimizer.apply_gradients(compute_gradients)
      minimize = optimizer.minimize(loss)
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

      ParameterServerModel.__init__(self, x, y_, compute_gradients, apply_gradients, minimize, accuracy, session)

