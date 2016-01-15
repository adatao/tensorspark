import tornado.web
import tornado.ioloop
import tornado.websocket
import json
import numpy as np
import tensorflow as tf
from threading import Lock
import tensorsparkmodel
import pickle
import time

class ParameterWebsocketServer(tornado.websocket.WebSocketHandler):
   def open(self):
      print "New client connected"

   def on_close(self):
      print "Client disconnected"

   def on_message(self, message):
      message = json.loads(message)
      print 'received message %s' % message['type']
      if message['type'] == 'client_requests_parameters':
         lock.acquire()
         parameters = model.get_parameters()
         lock.release()
         print 'sending parameters'
         parameters = pickle.dumps(parameters)
         self.write_message(parameters)
      elif message['type'] == 'client_gives_gradient':
         # deserialize!
         gradient = pickle.loads(message['gradient'])
         print 'received gradient'
         lock.acquire()
         model.apply_gradients(gradient)
         lock.release()
         print 'applied gradient'
      elif message['type'] == 'save_model':
         saver.save(model.session, 'parameter_server_model', global_step=int(time.time()))
      elif message['type'] == 'restore_model':
         model_path = message['model_path']
         saver.restore(model.session, model_path)
      else:
         print 'Unknown message type %s' % message['type']


model = tensorsparkmodel.TensorSparkModel()
saver = tf.train.Saver()
lock = Lock()
application = tornado.web.Application([
  (r"/", ParameterWebsocketServer),
  ])

if __name__ == "__main__":
  application.listen(55555)
  tornado.ioloop.IOLoop.current().start()
  print 'started'