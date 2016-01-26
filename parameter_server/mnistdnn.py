import tensorflow as tf
import numpy as numpy
from parameterservermodel import ParameterServerModel

def weight_variable(shape, name):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)


# 6 layers, 1024, 512 neurons

class MnistDNN(ParameterServerModel):
	def __init__(self):
		session = tf.InteractiveSession()
		x = tf.placeholder("float", shape=[None, 784], name='x')
		#x_image = tf.reshape(x, [-1,28,28,1], name='reshape')
		true_y = tf.placeholder("float", shape=[None, 10], name='y_')

		W_fc0 = weight_variable([784, 1024], 'W_fc0')
		b_fc0 = bias_variable([1024], 'b_fc0')
		h_fc0 = tf.nn.relu(tf.matmul(x, W_fc0) + b_fc0)

		W_fc1 = weight_variable([1024, 512], 'W_fc1')
		b_fc1 = bias_variable([512], 'b_fc1')
		h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

		W_fc2 = weight_variable([512, 1024], 'W_fc2')
		b_fc2 = bias_variable([1024], 'b_fc2')
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		W_fc3 = weight_variable([1024, 512], 'W_fc3')
		b_fc3 = bias_variable([512], 'b_fc3')
		h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

		W_fc4 = weight_variable([512, 1024], 'W_fc4')
		b_fc4 = bias_variable([1024], 'b_fc4')
		h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

		W_fc5 = weight_variable([1024, 512], 'W_fc5')
		b_fc5 = bias_variable([512], 'b_fc5')
		h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

		W_fc6 = weight_variable([512, 1024], 'W_fc6')
		b_fc6 = bias_variable([1024], 'b_fc6')
		h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

		W_fc7 = weight_variable([1024, 10], 'W_fc7')
		b_fc7 = bias_variable([10], 'b_fc7')

		keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)
		h_fc6_dropout = tf.nn.dropout(h_fc6, keep_prob)

		guess_y = tf.matmul(h_fc6, W_fc7) + b_fc7
		guess_y_dropout = tf.matmul(h_fc6_dropout, W_fc7) + b_fc7

		variables = [W_fc0, b_fc0, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4, W_fc5, b_fc5, W_fc6, b_fc6, W_fc7, b_fc7]
		loss = tf.nn.softmax_cross_entropy_with_logits(guess_y_dropout, true_y)

		optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
		compute_gradients = optimizer.compute_gradients(loss, variables)
		apply_gradients = optimizer.apply_gradients(compute_gradients)
		minimize = optimizer.minimize(loss)
		correct_prediction = tf.equal(tf.argmax(guess_y,1), tf.argmax(true_y,1))
		error_rate = 1 - tf.reduce_mean(tf.cast(correct_prediction, "float"))

		ParameterServerModel.__init__(self, x, true_y, compute_gradients, apply_gradients, minimize, error_rate, session)

	def process_warmup_data(self, data, batch_size=0):
	   num_classes = self.get_num_classes()
	   features = []
	   labels = []
	   if batch_size == 0:
	      batch_size = len(data)
	   for line in data:
	      if len(line) is 0:
	         print 'Skipping empty line'
	         continue
	      label = [0] * num_classes
	      split = line.split(',')
	      split[0] = int(split[0])
	      if split[0] >= num_classes:
	         print 'Error label out of range: %d' % split[0]
	         continue
	      features.append(split[1:])
	      label[split[0]] = 1
	      labels.append(label)

	   return labels, features

	def process_partition(self, partition, batch_size=0):
		num_classes = self.get_num_classes()
		features = []
		labels = []
		if batch_size == 0:
			batch_size = 1000000
		for i in xrange(batch_size):
		 try:
		    line = partition.next()
		    if len(line) is 0:
		       print 'Skipping empty line'
		       continue
		    label = [0] * num_classes
		    split = line.split(',')
		    split[0] = int(split[0])
		    if split[0] >= num_classes:
		       print 'Error label out of range: %d' % split[0]
		       continue
		    features.append(split[1:])
		    label[split[0]] = 1
		    labels.append(label)
		 except StopIteration:
		    break

		return labels, features


