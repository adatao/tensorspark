import tensorflow as tf
import numpy as np
import time
import json
import cStringIO
import base64
#from memory_profiler import profile
#import sys

#mod (by default, the error log is not written on local disk where the Driver is running)
isWritingErrorLogOnLocaldisk = False
# (set the same flag in tensorspark.py)

class ParameterServerModel():

   def __init__(self, x, y_, compute_gradients, apply_gradients, minimize, error_rate, session, batch_size):
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
            #mod:            
            if isWritingErrorLogOnLocaldisk == True:                                                                                                                                                
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
                                                                                                                                                                           
   def serialize(self, array):                                                                                                                                             
        return array.dumps()                                                                                                                                               
                                                                                                                                                                           
   def deserialize(self, serialized):                                                                                                                                      
        return np.loads(serialized)                                                                                                                                        
                                                                                                                                                                           
