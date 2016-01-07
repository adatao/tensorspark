import tornado.web
import tornado.ioloop
import tornado.websocket
import json
import numpy as np
import tensorflow as tf
from threading import Lock
import tensorsparkmodel
import pickle

class ParameterWebsocketServer(tornado.websocket.WebSocketHandler):
   def open(self):
      print "New client connected"

   def on_close(self):
      print "Client disconnected"

   def on_message(self, message):
      message = json.loads(message)
      if message['type'] == 'client_requests_parameters':
         lock.acquire()
         parameters = model.get_parameters()
         lock.release()
         print 'sending %d parameters ' % len(parameters)
         parameters = pickle.dumps(parameters) #, ensure_ascii=False, encoding='latin-1')
         self.write_message(parameters)
      elif message['type'] == 'client_gives_gradient':
         # deserialize!
         gradient = pickle.loads(message['gradient'])
         print 'received gradient'
         lock.acquire()
         self.model.apply_gradients(gradient)
         lock.release()
         print 'applied gradient'
      else:
         print 'Unknown message type %s' % message['type']


model = tensorsparkmodel.TensorSparkModel()
lock = Lock()
application = tornado.web.Application([
    (r"/", ParameterWebsocketServer),
])

if __name__ == "__main__":
    application.listen(55555)
    tornado.ioloop.IOLoop.current().start()
