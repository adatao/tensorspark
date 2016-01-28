#!/usr/local/bin/bpython
# -*- coding: utf-8 -*-
#import websocket
import json
import tensorflow as tf
#from tensorsparkmodel import TensorSparkModel
#from mnistcnn import MnistCNN
import mnistdnn
import higgsdnn
import moleculardnn
#import higgsdnn
#import download_mnist
import pickle
import math
import tornado.websocket
from tornado import gen 
from tornado.ioloop import IOLoop

# TODO
# Imagenet
# Tachyon
# Xavier initialization

class TensorSparkWorker():

   def __init__(self):
      self.model = mnistdnn.MnistDNN()      
#      self.model = moleculardnn.MolecularDNN()
#      self.model = higgsdnn.HiggsDNN()      
#      self.websock = websocket.create_connection('ws://localhost:55555')
      IOLoop.current().run_sync(self.init_websocket)
      self.iteration = 0

   @gen.coroutine
   def init_websocket(self):
      self.websock = yield tornado.websocket.websocket_connect("ws://localhost:55555/")

   def train_partition(self, partition): 
      batch_size = 100
      accuracies = []
      while True:
         print 'TensorSparkWorker().train_partition iteration %d' % self.iteration
         labels, features = self.model.process_partition(partition, batch_size)

         if len(labels) is 0:
            break

         if self.time_to_pull(self.iteration):
            self.request_parameters()

         accuracy = self.model.train(labels, features)
         accuracies.append(accuracy)
         self.iteration += 1

         if self.time_to_push(self.iteration):
            self.push_gradients()

    #  No need to return anything meaningful here as asynchronous anyway
      return []
      #return [self.train(x) for x in partition]

   def test_partition(self, partition):
      labels, features = self.model.process_partition(partition)
      self.request_parameters()
      accuracy = self.model.test(labels, features)
      return [accuracy]      
      #return [self.test(x) for x in partition]

   def test(self, data):
      print 'TensorSparkWorker().test "%s"' % data
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
      print 'requesting parameters'
      pickled_parameters = yield self.websock.read_message()
      parameters = pickle.loads(pickled_parameters)
      print 'received parameters'
      self.model.assign_parameters(parameters)

   def push_gradients(self):
      IOLoop.current().run_sync(self.push_gradients_coroutine)

   @gen.coroutine
   def push_gradients_coroutine(self):
      print 'pushing gradients'
      gradient = pickle.dumps(self.model.get_gradients())
      gradient_update_message = {'type':'client_gives_gradient', 'gradient':gradient}
      self.websock.write_message(json.dumps(gradient_update_message))
      print 'pushed gradients'      

