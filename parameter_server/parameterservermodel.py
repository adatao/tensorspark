import tensorflow as tf
import numpy as np

class ParameterServerModel():

   def __init__(self, x, y_, compute_gradients, apply_gradients, accuracy, session, placeholder_gradients):
      if type(x) != tf.python.framework.ops.Tensor:
         raise(TypeError('x must be of type tf.python.framework.ops.Tensor'))
      if type(y_) != tf.python.framework.ops.Tensor:
         raise(TypeError('y_ must be of type tf.python.framework.ops.Tensor'))
      if type(accuracy) != tf.python.framework.ops.Tensor:
         raise(TypeError('accuracy must be of type tf.python.framework.ops.Tensor'))
      if self.compute_gradients_correct_type(compute_gradients) == False:
         raise(TypeError('compute_gradients must be a list of tuples of type (tf.python.framework.ops.Tensor,tf.python.ops.variables.Variable)'))
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
      self.reset_gradients()
      self.gradient_counter = tf.Variable(initial_value=0, trainable=False)
      self.placeholder_gradients = placeholder_gradients
      self.session.run(tf.initialize_all_variables())

   def compute_gradients_correct_type(self, compute_gradients):
      if type(compute_gradients) != list:
         return False
      for element in compute_gradients:
         if type(element) != tuple or len(element) != 2:
            return False
         if type(element[0]) != tf.python.framework.ops.Tensor or type(element[1]) != tf.python.ops.tensorflow.variables.Variable:
            return False
      return True


   def get_num_classes(self):
      return self.y_.get_shape().as_list()[1]

   def train(self, labels, features):
      feed_dict={self.x: features, self.y_: labels, self.keep_prob: 0.5}
      self.num_gradients += 1
      # this can probably be made more efficiently with tf.gradients or tf.add
      self.gradients = np.add(self.gradients, [grad_var[0].eval(feed_dict=feed_dict) for grad_var in self.compute_gradients])
      return self.accuracy.eval(feed_dict=feed_dict)

   def test(self, labels, features):
      feed_dict = {self.x: features, self.y_: labels, self.keep_prob: 1.0}
      test_accuracy = self.accuracy.eval(feed_dict=feed_dict)
      print 'accuracy %s' % test_accuracy
      return test_accuracy

   def get_parameters(self):
      return [grad_var[1].eval(session=self.session) for grad_var in self.compute_gradients]

   def assign_parameters(self, parameters):
      self.reset_gradients()
      for i, grad_var in enumerate(self.compute_gradients):
         variable = grad_var[1]
         parameter = parameters[i]
         variable.assign(parameters[i])

   def apply(self, gradients):
      with self.graph.as_default():
         grads_and_vars = [(tf.convert_to_tensor(gradient), self.compute_gradients[i][1]) for i, gradient in enumerate(gradients)]
         feed_dict = {}
         for i, grad_var in enumerate(self.compute_gradients):
#            name = 'placeholder_%s' % grad_var[1].name
#            feed_dict[self.placeholder_gradients[i][0]] = tf.convert_to_tensor(gradients[i])
            feed_dict[self.placeholder_gradients[i][0]] = gradients[i]
#            placeholder_gradients.append((tf.placeholder('float', shape=grad_var[1].get_shape(), name='placeholder_%s' % name) ,grad_var[1]))


#         for i, grad_var in enumerate(self.compute_gradients):
#            feed_dict[grad_var[0]] = grads_and_vars[i][0]
#         feed_dict = { self.compute_gradients: grads_and_vars}
         print 'applying gradients %s' % grads_and_vars
         #apply_gradients = self.optimizer.apply_gradients(grads_and_vars) #, global_step=self.gradient_counter)
         #print 'graph for apply_gradients'
         #for op in apply_gradients.graph.get_operations():
         #   print op.name        
         #print 'graph for session'
         #for op in self.session.graph.get_operations():
         #   print op.name
         self.apply_gradients.run(session=self.session, feed_dict=feed_dict)
         print 'applied gradients'

   def get_gradients(self):
      return [np.divide(gradient,self.num_gradients).astype('float32') for gradient in self.gradients] 

   def reset_gradients(self):
      self.gradients = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]
      self.num_gradients = 0


