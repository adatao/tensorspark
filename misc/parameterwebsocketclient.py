import websocket
import json
import tensorflow as tf
import numpy as np
from tensorsparkmodel import TensorSparkModel
import download_mnist
import pickle


class TensorSparkWorker():
   def __init__(self):
      self.model = TensorSparkModel()

   def run(self):
      websock = websocket.create_connection('ws://localhost:55555')
      iterations_per_push = 100
      iterations_per_pull = 100
      for iteration in range(20000):
         if iteration % iterations_per_pull + 1 == 0:
            request_model_message = {'type':'client_requests_parameters'}
            websock.send(json.dumps(request_model_message))
            print 'requesting parameters'
            parameters = pickle.loads(websock.recv())
            print 'received %d parameters' % (len(parameters))

            #parameters = [np.fromstring(param, dtype=np.float32) for param in parameters]

            self.model.assign_parameters(parameters)

         self.model.train()

         if iteration % iterations_per_push == 0:
            print 'pushing'
            gradient = pickle.dumps(self.model.get_gradients())
            print 'got gradients'
            gradient_update_message = {'type':'client_gives_gradient', 'gradient':gradient}
            websock.send(json.dumps(gradient_update_message))


worker = TensorSparkWorker()
worker.run()
