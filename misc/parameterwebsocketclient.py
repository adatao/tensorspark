import websocket
import json
import tensorflow as tf
import download_mnist

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

class TensorSparkWorker():
   def __init__(self):
      self.model = None
      self.gradients = None
      # build model here
      self.mnist = download_mnist.read_data_sets('MNIST_data', one_hot=True)
      self.session = tf.InteractiveSession()
      x = tf.placeholder("float", shape=[None, 784])
      x_image = tf.reshape(x, [-1,28,28,1])
      y_ = tf.placeholder("float", shape=[None, 10])
      W_conv1 = weight_variable([5, 5, 1, 32])
      b_conv1 = bias_variable([32])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)

      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      b_fc1 = bias_variable([1024])

      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      keep_prob = tf.placeholder("float")
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])

      y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
      y = tf.nn.softmax(tf.matmul(x,W) + b)

      self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
      self.loss = -tf.reduce_sum(y_*tf.log(y_conv))
      self.optimizer = tf.train.AdamOptimizer(1e-4) #.minimize(cross_entropy)
      self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      self.session.run(tf.initialize_all_variables())

   def train(self, model):
      batch = self.mnist.train.next_batch(50)
      new_gradients, variables = self.optimizer.compute_gradients(self.loss, self.variables)
      self.gradients += new_gradients


   def run(self):
      websock = websocket.create_connection('ws://localhost:55555')
      iteration = 0
      iterations_per_push = 100
      iterations_per_pull = 100
      for iteration in range(20000):
         if iteration % iterations_per_pull == 0:
            request_model_message = {'type':'client_requests_model'}
            websock.send(json.dumps(request_model_message))
            parameters = websock.recv()


         self.train()


         if iteration % iterations_per_push == 0:
            gradient_update_message = {'type':'client_gives_gradient', 'gradient':self.gradients/iterations_per_push}
            websock.send(json.dumps(gradient_update_message))


worker = TensorSparkWorker()
worker.run()
