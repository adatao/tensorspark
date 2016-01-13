#import download_mnist
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
#      self.mnist = download_mnist.read_data_sets('MNIST_data', one_hot=True)
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
      self.loss = -tf.reduce_sum(self.y_ * tf.log(y_conv))
      self.optimizer = tf.train.AdamOptimizer(1e-4)
      self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
      self.compute_gradients = self.optimizer.compute_gradients(self.loss, self.variables)
      #print 'compute_gradients %s' % self.compute_gradients
      self.reset_gradients()
#      self.grads_and_vars = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]
#      for var in self.variables:
#         print var.get_shape()
#      print 
#      for grad in self.compute_gradients:
#         print grad[0].eval().get_shape()
#      print [var[0].get_shape() for var in self.compute_gradients]
#      self.gradient_holders = [(g[0],tf.Variable(tf.zeros(g[1].get_shape()))) for i, g in enumerate(self.compute_gradients)]
#      self.apply_gradients = self.optimizer.apply_gradients(self.gradient_holders)
      self.session.run(tf.initialize_all_variables())

   def train(self, data):
      print 'TensorSparkModel().train %s' % data
      #batch = self.mnist.train.next_batch(1000)
      split = data.split(',')
      label = [[0] * 10]
      label[0][int(split[0])] = 1.
      training_data = [split[1:]]
#      for datum in data:
#         split = datum.split(',')
#         labels.append(int(split[0]))
#         training_data.append([int(x) for x in split[1:]])

      feed_dict={self.x: training_data, self.y_: label, self.keep_prob: 0.5}

      self.num_gradients += 1
#      print 'compute_gradients: %s' % self.compute_gradients
      #self.apply_gradients.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
      self.gradients = np.add(self.gradients, [grad_var[0].eval(feed_dict=feed_dict) for grad_var in self.compute_gradients])
#      self.gradients = [np.add(self.gradients[i], grad_var[0].eval(feed_dict=feed_dict)) for i, grad_var in enumerate(self.compute_gradients)]
      print 'completed training iteration'
      #return self.accuracy.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
      return self.accuracy.eval(feed_dict=feed_dict)

   def test(self, data):
#      labels = []
#      testing_data = []
#      for datum in data:
#         split = datum.split(',')
#         labels.append[split[0]]
#         testing_data.append([split[1:]])
      print 'TensorSparkModel().test %s' % data
      split = data.split(',')
      label = [[0] * 10]
      label[0][int(split[0])] = 1.
      testing_data = [split[1:]]

      feed_dict = {self.x: testing_data, self.y_: label, self.keep_prob: 1.0}
      test_accuracy = self.accuracy.eval(feed_dict=feed_dict)
      print 'accuracy %s' % test_accuracy
      return test_accuracy

   def get_parameters(self):
      return [var.eval() for var in self.variables]

   def assign_parameters(self, parameters):
      self.reset_gradients()
      for i, variable in enumerate(self.variables):
         parameter = parameters[i]
         variable.assign(parameters[i])

   def apply_gradients(self, gradients):
      grads_and_vars = [(tf.convert_to_tensor(gradient), self.variables[i]) for i, gradient in enumerate(gradients)]
      self.optimizer.apply_gradients(grads_and_vars)

   def get_gradients(self):
#      return self.grads_and_vars
      return [np.divide(gradient,self.num_gradients).astype('float32') for gradient in self.gradients] 
#      feed_dict = { self.x: tf.zeros([784]), self.y_: tf.zeros([10]), self.keep_prob: 1.0 }
#      return [(grad[0].eval(feed_dict=feed_dict), grad[1]) for grad in self.gradient_holders]
#      for i, grad in enumerate(self.gradient_holders):
#         print i
#         fill = tf.fill(self.compute_gradients[i][1].get_shape(), self.num_gradients)
#         print fill.eval()
#         print 'fill shape %s grad shape %s' % (fill.get_shape(), grad[0].get_shape())
#         div  = tf.div(grad[0], fill)#.eval(feed_dict=feed_dict)
#         print div
#         print div.eval(feed_dict=feed_dict)
#      return [tf.div(grad[0], tf.fill(self.compute_gradients[i][0].get_shape(), self.num_gradients)).eval(feed_dict=feed_dict) for i, grad in enumerate(self.gradient_holders)]

   def reset_gradients(self):
      self.gradients = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]
      self.num_gradients = 0
#      self.gradients = [tf.Variable(tf.zeros(var.get_shape())) for var in self.variables]


