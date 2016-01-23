import tensorflow as tf
import numpy as np

class ParameterServerModel():

   def __init__(self, x, y_, compute_gradients, apply_gradients, minimize, accuracy, session):
      if type(x) != tf.python.framework.ops.Tensor:
         raise(TypeError('x must be of type tf.python.framework.ops.Tensor'))
      if type(y_) != tf.python.framework.ops.Tensor:
         raise(TypeError('y_ must be of type tf.python.framework.ops.Tensor'))
      if type(accuracy) != tf.python.framework.ops.Tensor:
         raise(TypeError('accuracy must be of type tf.python.framework.ops.Tensor'))
      if self.compute_gradients_is_correct_type(compute_gradients) == False:
         raise(TypeError('compute_gradients must be a list of tuples of type (tf.python.framework.ops.Tensor,tf.python.ops.Variable)'))
      if type(apply_gradients) != tf.python.framework.ops.Operation:
         raise(TypeError('apply_gradients must be of type tf.python.framework.ops.Operation'))
#      if type(optimizer) != tf.python.training.adam.AdamOptimizer:
#         raise(TypeError('optimizer must be of type tf.python.training.adam.AdamOptimizer'))
      if type(session) != tf.python.client.session.InteractiveSession:
         raise(TypeError('session must be of type tf.python.client.session.InteractiveSession'))

      self.session = session
      self.graph = session.graph
      self.session.graph.as_default().__enter__()
      self.x = x
      self.y_ = y_
      self.compute_gradients = compute_gradients
      self.apply_gradients = apply_gradients
      self.accuracy = accuracy
      self.accuracy_summary = tf.scalar_summary("accuracy", accuracy)
      self.minimize = minimize
      self.reset_gradients()
      self.gradient_counter = tf.Variable(initial_value=0, trainable=False)

      self.parameter_assignments = []
      for grad_var in self.compute_gradients:
         gradient = grad_var[0]
         variable = grad_var[1]
         self.parameter_assignments.append(variable.assign(gradient))


      self.merged = tf.merge_all_summaries()
      self.writer = tf.train.SummaryWriter("./logs", self.session.graph_def)
      self.session.run(tf.initialize_all_variables())

   def compute_gradients_is_correct_type(self, compute_gradients):
      if type(compute_gradients) != list:
         return False
      for element in compute_gradients:
         if type(element) != tuple or len(element) != 2:
            return False
         if type(element[0]) != tf.python.framework.ops.Tensor or type(element[1]) != tf.python.ops.tensorflow.Variable:
            return False
      return True

   def get_num_classes(self):
      return self.y_.get_shape().as_list()[1]

   def train(self, labels, features):
      feed_dict={self.x: features, self.y_: labels}
      # this can probably be made more efficiently with tf.gradients or tf.add
      self.gradients = np.add(self.gradients, [grad_var[0].eval(feed_dict=feed_dict) for grad_var in self.compute_gradients])
      #summary = self.merged.eval(feed_dict=feed_dict)
      accuracy = self.accuracy.eval(feed_dict=feed_dict)
#      summary, accuracy = self.session.run([self.merged, self.accuracy], feed_dict=feed_dict)
      #accuracy = self.accuracy.eval(feed_dict=feed_dict)
#      self.writer.add_summary(summary, self.num_gradients)

      self.num_gradients += 1
      return accuracy

   def test(self, labels, features):
      feed_dict = {self.x: features, self.y_: labels}
      test_accuracy = self.accuracy.eval(feed_dict=feed_dict)
      print 'accuracy %s' % test_accuracy
      return test_accuracy

   def get_parameters(self):
      return [grad_var[1].eval(session=self.session) for grad_var in self.compute_gradients]

   def assign_parameters(self, parameters):
      self.reset_gradients()
      for i, grad_var in enumerate(self.compute_gradients):
         gradient = grad_var[0]
         parameter = parameters[i]
         feed={gradient:parameter}
         self.parameter_assignments[i].eval(feed_dict=feed)
#         variable.assign(parameter).eval()

   def apply(self, gradients):
      with self.graph.as_default():
         grads_and_vars = [(tf.convert_to_tensor(gradient), self.compute_gradients[i][1]) for i, gradient in enumerate(gradients)]
         feed_dict = {}
         for i, grad_var in enumerate(self.compute_gradients):
            feed_dict[grad_var[0]] = gradients[i]


         self.apply_gradients.run(session=self.session, feed_dict=feed_dict)
         print 'applied gradients'

   def get_gradients(self):
      return [np.divide(gradient,self.num_gradients).astype('float32') for gradient in self.gradients] 

   def reset_gradients(self):
      self.gradients = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]
      self.num_gradients = 0


   def train_warmup(self, partition, batch_size=100): 
      accuracies = []
      iteration = 0
      while True:
         labels, features = process_warmup_data(self, partition, batch_size)

         if len(labels) is 0:
            break
         with self.session.as_default():
            #accuracy = self.train(labels, features)
            feed = {self.x: features, self.y_: labels}
            self.minimize.run(feed_dict = feed)
            accuracy = self.accuracy.eval(feed_dict=feed)
            accuracies.append(accuracy)
            iteration += 1
            print 'Warmup training iteration %d at %f accuracy' % (iteration, accuracy)

      return accuracies

# pulled this out of the class so we can do static vars via attaching properties to the function
def process_warmup_data(model, partition, batch_size=0):
   if 'next_index' not in process_warmup_data.__dict__:
      process_warmup_data.next_index = 0
   num_classes = model.get_num_classes()
   features = []
   labels = []
   if batch_size == 0:
      batch_size = len(partition)
   last_index = min(len(partition), process_warmup_data.next_index + batch_size)
   for i in range(process_warmup_data.next_index, last_index):
      line = partition[i]
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

   process_warmup_data.next_index = last_index
   return labels, features

def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)
