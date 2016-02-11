import tensorflow as tf
import numpy as np
import time
import json
import cStringIO
import base64
#from memory_profiler import profile
#import sys

class ParameterServerModel():

   def __init__(self, x, y_, compute_gradients, apply_gradients, minimize, error_rate, session, batch_size):
#       if type(x) != tf.python.framework.ops.Tensor:
#          raise(TypeError('x must be of type tf.python.framework.ops.Tensor'))
#       if type(y_) != tf.python.framework.ops.Tensor:
#          raise(TypeError('y_ must be of type tf.python.framework.ops.Tensor'))
#       if type(accuracy) != tf.python.framework.ops.Tensor:
#          raise(TypeError('accuracy must be of type tf.python.framework.ops.Tensor'))
#       if self.compute_gradients_is_correct_type(compute_gradients) == False:
#          raise(TypeError('compute_gradients must be a list of tuples of type (tf.python.framework.ops.Tensor,tf.python.ops.Variable)'))
#       if type(apply_gradients) != tf.python.framework.ops.Operation:
#          raise(TypeError('apply_gradients must be of type tf.python.framework.ops.Operation'))
# #      if type(optimizer) != tf.python.training.adam.AdamOptimizer:
# #         raise(TypeError('optimizer must be of type tf.python.training.adam.AdamOptimizer'))
#       if type(session) != tf.python.client.session.InteractiveSession:
#          raise(TypeError('session must be of type tf.python.client.session.InteractiveSession'))
      self.session = session
      self.batch_size = batch_size
      self.graph = session.graph
      self.session.graph.as_default().__enter__()
      self.x = x
      self.y_ = y_
      self.compute_gradients = compute_gradients
      self.apply_gradients = apply_gradients
      self.error_rate = error_rate
      self.error_rate_summary = tf.scalar_summary("error_rate", error_rate)
      self.minimize = minimize
      self.reset_gradients()
      self.gradient_counter = tf.Variable(initial_value=0, trainable=False)

#      self.parameter_assignments = []
#      for grad_var in self.compute_gradients:
#         gradient = grad_var[0]
#         variable = grad_var[1]
#         self.parameter_assignments.append(variable.assign(gradient))

      self.parameter_assignments = [None]*len(self.compute_gradients)
      for i in xrange(len(self.compute_gradients)):
          gradient = self.compute_gradients[i][0]
          variable = self.compute_gradients[i][1]
          self.parameter_assignments[i] = variable.assign(gradient)
      self.session.run(tf.initialize_all_variables())

   def get_num_classes(self):                                                                                                                                              
      return self.y_.get_shape().as_list()[1]                                                                                                                              
                                                                                                                                                                           
   def train(self, labels, features):                                                                                                                                      
      with self.session.as_default():                                                                                                                                      
                                                                                                                                                                           
        feed = {self.x: features, self.y_: labels}                                                                                                                         
        for i in range(len(self.compute_gradients)):                                                                                                                       
                self.gradients[i] += self.compute_gradients[i][0].eval(feed_dict=feed)                                                                                     
                                                                                                                                                                           
                                                                                                                                                                           
                                                                                                                                                                           
         # this can probably be made more efficiently with tf.gradients or tf.add                                                                                          
         #plus_this = [grad_var[0].eval(feed_dict=feed_dict) for grad_var in self.compute_gradients]                                                                       
         #self.gradients = np.add(self.gradients, plus_this)                                                                                                               
         #summary = self.merged.eval(feed_dict=feed_dict)                                                                                                                  
         #error_rate = self.error_rate.eval(feed_dict=feed_dict)                                                                                                           
   #      summary, accuracy = self.session.run([self.merged, self.accuracy], feed_dict=feed_dict)                                                                          
         #accuracy = self.accuracy.eval(feed_dict=feed_dict)                                                                                                               
   #      self.writer.add_summary(summary, self.num_gradients)                                                                                                             
        self.num_gradients += 1                                                                                                                                            
        del feed                                                                                                                                                           
        #del plus_this                                                                                                                                                     
         #return error_rate                                                                                                                                                
                                                                                                                                                                           
   def test(self, labels, features):                                                                                                                                       
      with self.session.as_default():                                                                                                                                      
                                                                                                                                                                           
         feed = {self.x: features, self.y_: labels}                                                                                                                        
         test_error_rate = self.error_rate.eval(feed_dict=feed)                                                                                                            
         del feed                                                                                                                                                          
         return test_error_rate                                                                                                                                            
                                                                                                                                                                           
   #@profile(stream=sys.stdout)                                                                                                                                            
   def get_parameters(self):                                                                                                                                               
      with self.session.as_default():                                                                                                                                      
                                                                                                                                                                           
#       return np.array([grad_var[1].eval(session=self.session) for grad_var in self.compute_gradients])                                                                   
        result = [None]*len(self.compute_gradients)                                                                                                                        
        for i in xrange(len(self.compute_gradients)):                                                                                                                      
                result[i] = self.compute_gradients[i][1].eval(session=self.session)                                                                                        
        array = np.array(result)                                                                                                                                           
        del result[:]                                                                                                                                                      
        del result                                                                                                                                                         
        return array                                                                                                                                                       
        #return np.asarray(result)                                                                                                                                         
                                                                                                                                                                           
   def assign_parameters(self, parameters):                                                                                                                                
      with self.session.as_default():                                                                                                                                      
                                                                                                                                                                           
         self.reset_gradients()                                                                                                                                            
         for i, grad_var in enumerate(self.compute_gradients):                                                                                                             
            self.parameter_assignments[i].eval(feed_dict={grad_var[0]:parameters[i]})                                                                                      
                                                                                                                                                                           
   #@profile(stream=sys.stdout)                                                                                                                                            
   def apply(self, gradients):                                                                                                                                             
      with self.graph.as_default():                                                                                                                                        
         feed_dict = {}                                                                                                                                                    
         for i, grad_var in enumerate(self.compute_gradients):                                                                                                             
            feed_dict[grad_var[0]] = gradients[i]                                                                                                                          
                                                                                                                                                                           
                                                                                                                                                                           
         self.apply_gradients.run(session=self.session, feed_dict=feed_dict)                                                                                               
         del feed_dict                                                                                                                                                     
         del gradients                                                                                                                                                     
                                                                                                                                                                           
   def get_gradients(self):                                                                                                                                                
#      with self.session.as_default():                                                                                                                                     
        result = [None]*(1+len(self.gradients))                                                                                                                            
        for i in xrange(len(self.gradients)):                                                                                                                              
                result[i+1] = np.divide(self.gradients[i],self.num_gradients).astype('float32')                                                                            
        result[0] = [time.time()]                                                                                                                                          
        array = np.array(result)                                                                                                                                           
        del result[:]                                                                                                                                                      
        del result                                                                                                                                                         
        return array                                                                                                                                                       
        #return np.asarray(result)                                                                                                                                         
                                                                                                                                                                           
   #@profile(stream=sys.stdout)                                                                                                                                            
#   def get_gradients(self):                                                                                                                                               
#      with self.session.as_default():                                                                                                                                     
#       result = [None]*len(self.gradients)                                                                                                                                
#       for i in xrange(len(self.gradients)):                                                                                                                              
#               result[i] = np.divide(self.gradients[i],self.num_gradients).astype('float32')                                                                              
#       return np.asarray(result)                                                                                                                                          
        #l = [np.divide(gradient,self.num_gradients).astype('float32') for gradient in self.gradients]                                                                     
        #n = np.array(l)                                                                                                                                                   
        #del l                                                                                                                                                             
        #return n                                                                                                                                                          
      #return np.array([np.divide(gradient,self.num_gradients).astype('float32') for gradient in self.gradients])                                                          
                                                                                                                                                                           
   def reset_gradients(self):                                                                                                                                              
      with self.session.as_default():                                                                                                                                      
         self.gradients = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]                                                                              
         self.num_gradients = 0                                                                                                                                            
                                                                                                                                                                           
                                                                                                                                                                           
   def train_warmup(self, partition, error_rates_filename):                                                                                                                
      error_rates = []                                                                                                                                                     
      iteration = 0                                                                                                                                                        
      batch_size = self.batch_size                                                                                                                                         
      for i in range(0, len(partition), batch_size):                                                                                                                       
         data = partition[i:i+batch_size]                                                                                                                                  
         labels, features = self.process_data(data)                                                                                                                        
         if len(labels) is 0:                                                                                                                                              
            break                                                                                                                                                          
         with self.session.as_default():                                                                                                                                   
            #accuracy = self.train(labels, features)                                                                                                                       
            feed = {self.x: features, self.y_: labels}                                                                                                                     
            self.minimize.run(feed_dict = feed)                                                                                                                            
            error_rate = self.error_rate.eval(feed_dict=feed)                                                                                                              
            t = time.time()                                                                                                                                                
            with open(error_rates_filename, 'a') as f:                                                                                                                     
                f.write('%f , %f\n' % (t,error_rate))                                                                                                                      
            error_rates.append(error_rate)                                                                                                                                 
            iteration += 1                                                                                                                                                 
            print 'Warmup training iteration %d at %f error_rate' % (iteration, error_rate)                                                                                
                                                                                                                                                                           
      return error_rates                                                                                                                                                   
                                                                                                                                                                           
   def process_data(self, data):                                                                                                                                           
      raise AssertionError('function not implemented')                                                                                                                     
                                                                                                                                                                           
   def process_partition(self, partition):                                                                                                                                 
      raise AssertionError('function not implemented')                                                                                                                     
                                                                                                                                                                           
   #@profile(stream=sys.stdout)                                                                                                                                            
   def serialize(self, array):                                                                                                                                             
        return array.dumps()                                                                                                                                               
        #return json.dumps(thing)                                                                                                                                          
        #memfile = cStringIO.StringIO()                                                                                                                                    
        #np.savez_compressed(memfile,array=array)                                                                                                                          
        #np.savez(memfile,array=array)                                                                                                                                     
        #memfile.seek(0)                                                                                                                                                   
        #readed = base64.urlsafe_b64encode(memfile.read())                                                                                                                 
        #memfile.close()                                                                                                                                                   
        #del memfile                                                                                                                                                       
        #del array                                                                                                                                                         
        #return readed                                                                                                                                                     
                                                                                                                                                                           
   #@profile(stream=sys.stdout)                                                                                                                                            
   def deserialize(self, serialized):                                                                                                                                      
        return np.loads(serialized)                                                                                                                                        
                                                                                                                                                                           
