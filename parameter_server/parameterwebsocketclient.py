#!/usr/local/bin/bpython
# -*- coding: utf-8 -*-
import json
<<<<<<< HEAD
import tensorflow as tf
import mnistdnn
import higgsdnn
import moleculardnn
#import pickle
import math
import numpy as np
import tornado.websocket
from tornado import gen
from tornado.ioloop import IOLoop
import cStringIO
=======
import threading
import tensorflow as tf

import tornado.websocket
from tornado import gen 
from tornado.ioloop import IOLoop
import cStringIO

import mnistdnn
import higgsdnn
import moleculardnn

from memory_profiler import profile
import sys

#import gc

>>>>>>> 60488ef3cae794043766b647bcb88ef164650d88
# TODO
# Imagenet
# Tachyon
# Xavier initialization

class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
<<<<<<< HEAD


class TensorSparkWorker(Borg):

   def __init__(self, model_keyword, batch_size, websocket_port):
      Borg.__init__(self)

      if 'model' not in self.__dict__:
         print 'Creating new Borg worker'
         if model_keyword == 'mnist':
             self.model = mnistdnn.MnistDNN(batch_size)
         elif model_keyword == 'higgs':
             self.model = higgsdnn.HiggsDNN(batch_size)
         elif model_keyword == 'molecular':
             self.model = moleculardnn.MolecularDNN(batch_size)
         else:
            raise
         self.batch_size = batch_size
         self.websocket_port = websocket_port
         self.loop = IOLoop.current()
         self.loop.run_sync(self.init_websocket)
         self.iteration = 0




      #self.batch_size = batch_size
      #self.websocket_port = websocket_port
      #if model_keyword == 'mnist':
      #    self.model = mnistdnn.MnistDNN(batch_size)    
      #elif model_keyword == 'higgs':
      #    self.model = higgsdnn.HiggsDNN(batch_size)  
      #elif model_keyword == 'molecular':
      #    self.model = moleculardnn.MolecularDNN(batch_size)
      #else:
#         raise                                                                                                                                 
 #     IOLoop.current().run_sync(self.init_websocket)                                                                                           
 #     self.iteration = 0                                                                                                                       
                                                                                                                                                
   @gen.coroutine                                                                                                                               
   def init_websocket(self):                                                                                                                    
      self.websock = yield tornado.websocket.websocket_connect("ws://172.31.3.244:%d/" % self.websocket_port, connect_timeout=3600)             
                                                                                                                                                
   def train_partition(self, partition):                                                                                                        
      while True:                                                                                                                               
         #print 'TensorSparkWorker().train_partition iteration %d' % self.iteration                                                             
         labels, features = self.model.process_partition(partition)                                                                             
                                                                                                                                                
         if len(labels) is 0:                                                                                                                   
            break                                                                                                                               
                                                                                                                                                
         if self.time_to_pull(self.iteration):                                                                                                  
                self.request_parameters()                                                                                                       
                                                                                                                                                
         self.model.train(labels, features)                                                                                                     
         self.iteration += 1                                                                                                                    
                                                                                                                                                
         if self.time_to_push(self.iteration):                                                                                                  
            self.push_gradients()                                                                                                               
                                                                                                                                                
      return []                                                                                                                                 
      #return [self.train(x) for x in partition]                                                                                                
                                                                                                                                                
   def test_partition(self, partition):                                                                                                         
      labels, features = self.model.process_partition(partition)                                                                                
      self.request_parameters()                                                                                                                 
      error_rate = self.model.test(labels, features)                                                                                            
      return [error_rate]                                                                                                                       
      #return [self.test(x) for x in partition]                                                                                                 
                                                                                                                                                
   def test(self, data):                                                                                                                        
      #print 'TensorSparkWorker().test "%s"' % data                                                                                             
      if len(data) is 0:                                                                                                                        
         return 1.0                                                                                                                             
      self.request_parameters()                                                                                                                 
      accuracy = self.model.test(data)                                                                                                          
      return accuracy                                                                                                                           
#      self.model.                                                                                                                              
                                                                                                                                                
   def time_to_pull(self, iteration):                                                                                                           
      return iteration % 5 == 0                                                                                                                 
#      return True                                                                                                                              
                                                                                                                                                
   def time_to_push(self, iteration):                                                                                                           
      return iteration % 5 == 0                                                                                                                 
#      return True                                                                                                                              
                                                                                                                                                
   def request_parameters(self):                                                                                                                
      IOLoop.current().run_sync(self.request_parameters_coroutine)                                                                              
                                                                                                                                                
   @gen.coroutine                                                                                                                               
   def request_parameters_coroutine(self):                                                                                                      
      request_model_message = {'type':'client_requests_parameters'}                                                                             
      self.websock.write_message(json.dumps(request_model_message))                                                                             
      #print 'requesting parameters'                                                                                                            
      parameters = yield self.websock.read_message()                                                                                            
      parameters = self.model.deserialize(parameters)                                                                                           
      #parameters = pickle.loads(pickled_parameters)                                                                                            
      #print 'received parameters'                                                                                                              
      self.model.assign_parameters(parameters)                                                                                                  
                                                                                                                                                
   def push_gradients(self):                                                                                                                    
      IOLoop.current().run_sync(self.push_gradients_coroutine)                                                                                  
                                                                                                                                                
   @gen.coroutine                                                                                                                               
   def push_gradients_coroutine(self):                                                                                                          
        #print 'pushing gradients'                                                                                                              
        gradients = self.model.serialize(self.model.get_gradients())                                                                            
        gradient_update_message = {'type':'client_gives_gradient', 'gradient':gradients}                                                        
        self.websock.write_message(json.dumps(gradient_update_message))                                                                         
        #print 'pushed gradients'                                                                                                               
                                                                                                                                                
=======




class TensorSparkWorker(Borg):

#   @profile(stream=sys.stdout)
   def __init__(self, model_keyword, batch_size, websocket_port):
      Borg.__init__(self)

      if 'model' not in self.__dict__:
         print 'Creating new Borg worker'
         if model_keyword == 'mnist':
             self.model = mnistdnn.MnistDNN(batch_size)    
         elif model_keyword == 'higgs':
             self.model = higgsdnn.HiggsDNN(batch_size)  
         elif model_keyword == 'molecular':
             self.model = moleculardnn.MolecularDNN(batch_size)
         else:
            raise
         self.batch_size = batch_size
         self.websocket_port = websocket_port
         self.loop = IOLoop.current()
         self.loop.run_sync(self.init_websocket)
         self.lock = threading.Lock()
         self.iterations = 0

   @gen.coroutine
   def init_websocket(self):
      self.websock = yield tornado.websocket.websocket_connect("ws://localhost:%d/" % self.websocket_port)

#   @profile(stream=sys.stdout)
   def train_partition(self, partition): 
      while True:
         print 'TensorSparkWorker().train_partition iteration %d' % self.iterations
         labels, features = self.model.process_partition(partition)
         if len(labels) is 0:
            break

         if self.time_to_pull(self.iterations):
            self.request_parameters()


         self.lock.acquire()
         self.model.train(labels, features)
         self.lock.release()

         self.iterations += 1


         if self.time_to_push(self.iterations):
            self.push_gradients()



         del labels[:]
         del features[:]


      del partition


      return []

   def test_partition(self, partition):
      labels, features = self.model.process_partition(partition)
      self.request_parameters()
      error_rate = self.model.test(labels, features)
      del labels
      del features
      del partition
      return [error_rate]      

   def test(self, data):
      if len(data) is 0:
         return 1.0
      self.request_parameters()
      accuracy = self.model.test(data)
      return accuracy

   def time_to_pull(self, iteration):
      return iteration % 5 == 0 
#      return True

   def time_to_push(self, iteration):
      return iteration % 5 == 0
#      return True

#   @profile(stream=sys.stdout)
   def request_parameters(self):
      self.loop.run_sync(self.request_parameters_coroutine)

   @gen.coroutine
   def request_parameters_coroutine(self):

      request_model_message = {'type':'client_requests_parameters'}
      self.websock.write_message(json.dumps(request_model_message))
      parameters = yield self.websock.read_message()
      deserialized = self.model.deserialize(parameters)
      self.lock.acquire()
      self.model.assign_parameters(deserialized)
      self.lock.release()
      del parameters
      del deserialized

#   @profile(stream=sys.stdout)
   def push_gradients(self):
      self.loop.run_sync(self.push_gradients_coroutine)

   @gen.coroutine
   def push_gradients_coroutine(self):
      # print 'pushing gradients'
      self.lock.acquire()
      grads = self.model.get_gradients()
      self.lock.release()
      gradients = self.model.serialize(grads)
      gradient_update_message = {'type':'client_gives_gradient', 'gradient':gradients}
      self.websock.write_message(json.dumps(gradient_update_message))
      del grads
      del gradients
      gradient_update_message.clear()

>>>>>>> 60488ef3cae794043766b647bcb88ef164650d88
