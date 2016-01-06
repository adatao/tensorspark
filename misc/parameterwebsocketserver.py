import tornado.web
import tornado.ioloop
import tornado.websocket
import json
import numpy as np
import tensorflow as tf
from threading import Lock


class ParameterWebsocketServer(tornado.websocket.WebSocketHandler):
   def open(self):
      print "New client connected"

   def on_close(self):
      print "Client disconnected"

   def on_message(self, message):
      message = json.loads(message)
      if message['type'] == 'client_requests_model':
         lock.acquire()
         print 'sending model'
         self.write_message(model)
         lock.release()
      elif message['type'] == 'client_gives_gradient':
         lock.acquire()
         #self.model.get_optimizer().apply_gradients(message.gradient)
         print 'received gradient %s' % message['gradient']
         lock.release()
      else:
         print 'Unknown message type %s' % message['type']


model = 'model'
lock = Lock()
application = tornado.web.Application([
    (r"/", ParameterWebsocketServer),
])

if __name__ == "__main__":
    application.listen(55555)
    tornado.ioloop.IOLoop.current().start()
