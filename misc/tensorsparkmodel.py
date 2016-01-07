import download_mnist
import tensorflow as tf
import numpy as np

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

class TensorSparkModel():

   def __init__(self):
      self.mnist = download_mnist.read_data_sets('MNIST_data', one_hot=True)
      self.session = tf.InteractiveSession()
      self.x = tf.placeholder("float", shape=[None, 784], name='x')
      x_image = tf.reshape(self.x, [-1,28,28,1], name='reshape')
      self.y_ = tf.placeholder("float", shape=[None, 10], name='y_')
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
      self.keep_prob = tf.placeholder("float", name='keep_prob')
      h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
      W_fc2 = weight_variable([1024, 10], 'W_fc2')
      b_fc2 = bias_variable([10], 'b_fc2')
      y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

      self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
      self.reset_gradients()
      self.loss = -tf.reduce_sum(self.y_ * tf.log(y_conv))
      self.optimizer = tf.train.AdamOptimizer(1e-4)
      self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
      self.compute_gradients = self.optimizer.compute_gradients(self.loss, self.variables)
      self.gradient_holders = [(g[0],self.gradients[i]) for i, g in enumerate(self.compute_gradients)]
      self.apply_gradients = self.optimizer.apply_gradients(self.gradient_holders)
      self.session.run(tf.initialize_all_variables())
      print [var[0].name for var in self.gradient_holders]

   def train(self):
      batch = self.mnist.train.next_batch(50)
      self.apply_gradients.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
      print 'completed training iteration'

   def test(self):
      test_accuracy = self.accuracy.eval(feed_dict=self.feed_dict)
      return test_accuracy

   def get_parameters(self):
      return [var.eval() for var in self.variables]

   def assign_parameters(self, parameters):
      self.reset_gradients()
      for i, variable in enumerate(self.variables):
         parameter = parameters[i]
         variable.assign(parameters[i])

   def apply_gradients(self, gradients):
      self.optimizer.apply_gradients(gradients)

   def get_gradients(self):
      feed_dict = { self.x: tf.zeros([784]), self.y_: tf.zeros([10]), self.keep_prob: 1.0 }
      for grad in self.gradient_holders:
         print grad[0].get_shape(), grad[1].name
         tf.div(grad[0], tf.fill(grad[0].get_shape(), self.num_gradients))
      return [tf.div(grad[0], tf.fill(grad[0].get_shape(), self.num_gradients)).eval() for grad in self.gradient_holders]

   def reset_gradients(self):
      self.num_gradients = 0.0
      for var in self.variables:
         print var.name
      self.gradients = [tf.Variable(tf.zeros(var.get_shape())) for var in self.variables]


