
import tensorflow as tf
from parameterservermodel import ParameterServerModel

def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)



class HiggsDNN(ParameterServerModel):
	def __init__(self):
		num_hidden_units = 1024
		session = tf.InteractiveSession()
		x = tf.placeholder("float", shape=[None, 28], name='x')
		true_y = tf.placeholder("float", shape=[None, 1], name='y_')

		W_fc1 = weight_variable([28, num_hidden_units])
		b_fc1 = bias_variable([num_hidden_units])
		h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
#		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		W_fc2 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc2 = bias_variable([num_hidden_units])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
#		h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

		W_fc3 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc3 = bias_variable([num_hidden_units])
		h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
#		h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

		W_fc4 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc4 = bias_variable([num_hidden_units])
		h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
#		h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

		W_fc5 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc5 = bias_variable([num_hidden_units])
		h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)
#		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

		W_fc6 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc6 = bias_variable([num_hidden_units])
		h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)
#		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

		W_fc7 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc7 = bias_variable([num_hidden_units])
		h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
#		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

		W_fc8 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc8 = bias_variable([num_hidden_units])
		h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)
#		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

		W_fc9 = weight_variable([num_hidden_units, num_hidden_units])
		b_fc9 = bias_variable([num_hidden_units])
		h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)
#		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

		W_fc10 = weight_variable([num_hidden_units, 1])
		b_fc10 = bias_variable([1])


		keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)
		h_fc9_dropout = tf.nn.dropout(h_fc9, keep_prob)

		guess_y = tf.matmul(h_fc9, W_fc10) + b_fc10
		guess_y_dropout = tf.matmul(h_fc9_dropout, W_fc10) + b_fc10

		variables = [W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4, W_fc5, b_fc5, W_fc6, b_fc6, W_fc7, b_fc7, W_fc8, b_fc8, W_fc9, b_fc9, W_fc10, b_fc10]
		#loss = tf.nn.softmax_cross_entropy_with_logits(guess_y_dropout, true_y)
		loss = tf.nn.l2_loss(guess_y_dropout - true_y)

#		optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
		optimizer = tf.train.RMSPropOptimizer(1e-4, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
		compute_gradients = optimizer.compute_gradients(loss, variables)
		apply_gradients = optimizer.apply_gradients(compute_gradients)
		minimize = optimizer.minimize(loss)
		correct_prediction = tf.equal(tf.clip_by_value(tf.round(guess_y), 0, 1), tf.clip_by_value(tf.round(true_y), 0, 1))
		error_rate = 1 - tf.reduce_mean(tf.cast(correct_prediction, "float"))
#		correct_prediction = tf.equal(tf.argmax(guess_y,1), tf.argmax(true_y,1))
#		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


		ParameterServerModel.__init__(self, x, true_y, compute_gradients, apply_gradients, minimize, error_rate, session)
