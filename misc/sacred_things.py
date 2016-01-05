import download_mnist
import tensorflow as tf

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('tensorflow001')
ex.observers.append(MongoObserver.create(db_name='sacred_experiments'))


mnist = download_mnist.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

@ex.config
def config_network():
   num_conv_layers = 2
   num_full_layers = 2
   filter_sizes = [5,5]
   num_features_per_layer = [32, 64, 1024, 10]
   num_iterations = 20000

@ex.capture
def setup_graph(num_conv_layers, num_full_layers, filter_sizes, num_features_per_layer, num_iterations):
   print('setting up graph')
   x = tf.placeholder("float", shape=[None, 784])
   y_ = tf.placeholder("float", shape=[None, 10])
   x_image = tf.reshape(x, [-1,28,28,1])

   W_conv = []
   b_conv = []
   h_conv = []
   h_pool = []
   for i in range(num_conv_layers):
      W_input_size = W_conv[i-1].get_shape().as_list()[3] if i > 0 else 1
      W_input = h_pool[i-1] if i > 0 else x_image
      W_shape = [filter_sizes[i], filter_sizes[i], W_input_size, num_features_per_layer[i]]

      W_conv.append(weight_variable(W_shape))
      b_conv.append(bias_variable([num_features_per_layer[i]]))
      h_conv.append(tf.nn.relu(conv2d(W_input, W_conv[i]) + b_conv[i]))
      h_pool.append(max_pool_2x2(h_conv[i]))

   last_pool_flat = tf.reshape(h_pool[-1], [-1, 7*7*num_features_per_layer[num_conv_layers-1]])

   W_fc = []
   b_fc = []
   h_fc = []
   h_fc_drop = []
   accuracy = None
   keep_prob = None
   for j in range(num_conv_layers, num_conv_layers + num_full_layers):
      i = j - num_conv_layers
      W_input_shape = [num_features_per_layer[j-1], num_features_per_layer[j]] if i > 0 else [7 * 7 * num_features_per_layer[j-1], num_features_per_layer[j]]
      W_input = h_fc_drop[i-1] if i > 0 else last_pool_flat
      W_fc.append(weight_variable(W_input_shape))
      b_fc.append(bias_variable([num_features_per_layer[j]]))
      if i < num_full_layers - 1:
         keep_prob = tf.placeholder("float")
         h_fc.append(tf.nn.relu(tf.matmul(W_input, W_fc[i]) + b_fc[i]))
         h_fc_drop.append(tf.nn.dropout(h_fc[i], keep_prob))
      else:
         y_conv = tf.nn.softmax(tf.matmul(h_fc_drop[i-1], W_fc[i]) + b_fc[i])
         cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
         train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
         correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
         sess.run(tf.initialize_all_variables())

   print 'running experiment with %d iterations' % num_iterations
   for i in range(num_iterations):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      if i%100 == 0:
         train_accuracy = accuracy.eval(feed_dict={
               x:batch[0], y_: batch[1], keep_prob: 1.0})
         print("step %d, training accuracy %g"%(i, train_accuracy))
   test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
   return test_accuracy



@ex.automain
def my_main():
   test_accuracy = setup_graph()
   return test_accuracy