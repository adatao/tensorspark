import tensorflow as tf
#import download_mnist


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

#mnist = download_mnist.read_data_sets('MNIST_data', one_hot=True)

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
keep_prob = tf.placeholder("float", name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10], 'W_fc2')
b_fc2 = bias_variable([10], 'b_fc2')
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
#loss = -tf.reduce_sum(y_ * tf.log(y_conv))
#optimizer = tf.train.AdamOptimizer(1e-4)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#compute_gradients = optimizer.compute_gradients(loss)

variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
loss = -tf.reduce_sum(y_ * tf.log(y_conv))
optimizer = tf.train.AdamOptimizer(1e-4)
compute_gradients = optimizer.compute_gradients(loss, variables)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
session.run(tf.initialize_all_variables())
ops = session.graph.get_operations()
for element in compute_gradients:
   print type(element[0]), type(element[1]), element[1].name
#saver = tf.train.Saver()
#saver.restore(session, './')


#for i in range(2):
#      batch = mnist.train.next_batch(1000)
#      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
#      print("step %d, training accuracy %g"%(i, train_accuracy))
#      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#gradient_holders = [(g[0],tf.Variable(tf.zeros(g[1].get_shape()))) for i, g in enumerate(compute_gradients)]
#apply_gradients = optimizer.apply_gradients(gradient_holders)


#session.run(tf.initialize_all_variables())


#batch = mnist.train.next_batch(50)
#apply_gradients.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}

#feed_dict = { x: tf.zeros([784]), y_: tf.zeros([10]), keep_prob: 1.0 }

#gradients = []
#for grad_var in compute_gradients:
#      grad = grad_var[0]
#      print grad.get_shape()
#      grad = grad.eval(feed_dict=feed_dict)
#      print grad.shape
#      var = grad_var[1]
#      gradients.append((grad, var))
