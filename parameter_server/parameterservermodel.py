import tensorflow as tf
import numpy as np

class ParameterServerModel():

   def __init__(self, x, y_, compute_gradients, accuracy, session):
      if type(x) != tf.python.framework.ops.Tensor:
         raise(TypeError('x must be of type tf.python.framework.ops.Tensor'))
      if type(y_) != tf.python.framework.ops.Tensor:
         raise(TypeError('y_ must be of type tf.python.framework.ops.Tensor'))
      if type(accuracy) != tf.python.framework.ops.Tensor:
         raise(TypeError('accuracy must be of type tf.python.framework.ops.Tensor'))
      if self.compute_gradients_correct_type(compute_gradients) == False:
         raise(TypeError('compute_gradients must be a list of tuples of type (tf.python.framework.ops.Tensor,tf.python.ops.variables.Variable)'))
      if type(session) != tf.python.client.session.InteractiveSession:
         raise(TypeError('session must be of type tf.python.client.session.InteractiveSession'))

      self.x = x
      self.y_ = y_
      self.compute_gradients = compute_gradients
      self.accuracy = accuracy
      self.session = session
      self.reset_gradients()
      self.session.run(tf.initialize_all_variables())

   def compute_gradients_correct_type(self, compute_gradients):
      if type(compute_gradients) != list:
         print 'not a list'
         return False
      for element in compute_gradients:
         if type(element) != tuple or len(element) != 2:
            print 'not a 2 -tuple'
            return False
         if type(element[0]) != tf.python.framework.ops.Tensor or type(element[1]) != tf.python.ops.tensorflow.variables.Variable:
            print 'not a tensor, variable pair'
            return False
      return True


   def get_num_classes(self):
      return self.y_.get_shape().as_list()[1]

   def train(self, labels, features):
      feed_dict={self.x: features, self.y_: labels, self.keep_prob: 0.5}
      self.num_gradients += 1
      self.gradients = np.add(self.gradients, [grad_var[0].eval(feed_dict=feed_dict) for grad_var in self.compute_gradients])
      return self.accuracy.eval(feed_dict=feed_dict)

   def test(self, labels, features):
      feed_dict = {self.x: features, self.y_: labels, self.keep_prob: 1.0}
      test_accuracy = self.accuracy.eval(feed_dict=feed_dict)
      print 'accuracy %s' % test_accuracy
      return test_accuracy

   def get_parameters(self):
      return [grad_var[1].eval() for grad_var in self.compute_gradients]

   def assign_parameters(self, parameters):
      self.reset_gradients()
      for i, grad_var in enumerate(self.compute_gradients):
         variable = grad_var[1]
         parameter = parameters[i]
         variable.assign(parameters[i])

   def apply_gradients(self, gradients):
      grads_and_vars = [(tf.convert_to_tensor(gradient), self.compute_gradients[i][1]) for i, gradient in enumerate(gradients)]
      self.optimizer.apply_gradients(grads_and_vars)

   def get_gradients(self):
      return [np.divide(gradient,self.num_gradients).astype('float32') for gradient in self.gradients] 

   def reset_gradients(self):
      self.gradients = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]
      self.num_gradients = 0
