import parameterwebsocketclient
import pyspark
from operator import add
import websocket
import json
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import mnistcnn
import tensorflow as tf
import pickle


class ParameterServer(threading.Thread):

	class ParameterServerWebsocketHandler(tornado.websocket.WebSocketHandler):

		def open(self):
			print "New client connected"

		def on_close(self):
			print "Client disconnected"

		def on_message(self, message):
			message = json.loads(message)
			print 'received message %s' % message['type']
			if message['type'] == 'client_requests_parameters':
				ParameterServer.lock.acquire()
				parameters = ParameterServer.model.get_parameters()
				ParameterServer.lock.release()
				print 'sending parameters'
				parameters = pickle.dumps(parameters)
				self.write_message(parameters)
			elif message['type'] == 'client_gives_gradient':
				# deserialize!
				gradient = pickle.loads(message['gradient'])
				print 'received gradient'
				ParameterServer.lock.acquire()
				ParameterServer.model.apply_gradients(gradient)
				ParameterServer.lock.release()
				print 'applied gradient'
			elif message['type'] == 'save_model':
				ParameterServer.saver.save(ParameterServer.model.session, './models/parameter_server_model', global_step=int(time.time()))
			elif message['type'] == 'restore_model':
				model_path = message['model_path']
				ParameterServer.saver.restore(ParameterServer.model.session, model_path)
			else:
				print 'Unknown message type %s' % message['type']	

	model = mnistcnn.MnistCNN()
	saver = tf.train.Saver()
	lock = threading.Lock()
	application = tornado.web.Application([(r"/", ParameterServerWebsocketHandler)])

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		self.application.listen(55555)
		print 'listening at port 55555' 
  		tornado.ioloop.IOLoop.current().start()
  	

def train_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().train_partition(partition)

def test_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().test_partition(partition)

# you can find the mnist csv files here http://pjreddie.com/projects/mnist-in-csv/
def train_epochs(num_epochs):
#	training_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/mnist_train.csv')
	training_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/medium_mnist_train.csv')
	for i in range(num_epochs):
		mapped_training = training_rdd.mapPartitions(train_partition)
		mapped_training.collect()
		training_rdd.repartition(training_rdd.getNumPartitions())

def test_all():
	testing_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/mnist_test.csv')
	mapped_testing = testing_rdd.mapPartitions(test_partition)
	return mapped_testing.reduce(add)/mapped_testing.getNumPartitions()

def save_model():
	message = {'type': 'save_model'}
	websock.send(json.dumps(message))

def start_parameter_server():
	parameter_server = ParameterServer()
	parameter_server.start()
	return parameter_server

parameter_server = start_parameter_server()
raw_input('Press enter to continue\n')
sc = pyspark.SparkContext()
websock = websocket.create_connection('ws://localhost:55555')
num_epochs = 3
train_epochs(num_epochs)
print 'Done training'
save_model()
print 'Testing now'
print test_all()

tornado.ioloop.IOLoop.current().stop()
