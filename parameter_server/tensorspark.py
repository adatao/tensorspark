import parameterwebsocketclient
import pyspark
from operator import add
import websocket
import json
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
#import mnistcnn
import mnistdnn
import higgsdnn
import moleculardnn
import tensorflow as tf
import pickle
import time
from sacred import Experiment
from sacred.observers import MongoObserver

directory = "/Users/ushnishde/Documents/TensorSpark/"

class ParameterServerWebsocketHandler(tornado.websocket.WebSocketHandler):

	def __init__(self, *args, **kwargs):
		self.model = kwargs.pop('model')
		with self.model.session.graph.as_default():
			self.saver = tf.train.Saver()
		self.lock = threading.Lock()		
		super(ParameterServerWebsocketHandler,self).__init__(*args, **kwargs)

	def open(self):
		print "New client connected"

	def on_close(self):
		print "Client disconnected"

	def on_message(self, message):
		message = json.loads(message)
		print 'received message %s' % message['type']
		if message['type'] == 'client_requests_parameters':
			self.lock.acquire()
			parameters = self.model.get_parameters()
			self.lock.release()
			print 'sending parameters'
			parameters = pickle.dumps(parameters)
			self.write_message(parameters)
		elif message['type'] == 'client_gives_gradient':
			# deserialize!
			gradient = pickle.loads(message['gradient'])
			print 'received gradient'
			self.lock.acquire()
			self.model.apply(gradient)
			self.lock.release()
			print 'applied gradient'
		elif message['type'] == 'save_model':
			if not os.path.exists('./models'):
				os.makedirs('./models')
			self.saver.save(self.model.session, './models/parameter_server_model', global_step=int(time.time()))
		elif message['type'] == 'restore_model':
			model_path = message['model_path']
			self.saver.restore(self.model.session, model_path)
		else:
			print 'Unknown message type %s' % message['type']

class ParameterServer(threading.Thread):

	def __init__(self, model, warmup_data=None):
		threading.Thread.__init__(self)
		self.model = model
		self.warmup(warmup_data)
		self.application = tornado.web.Application([(r"/", ParameterServerWebsocketHandler, {'model':model})])

	def warmup(self, data=None):
		if data is not None:
			self.model.train_warmup(data)

	def run(self):
		self.application.listen(55555)
		print 'listening at port 55555'
   		tornado.ioloop.IOLoop.current().start()

def train_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().train_partition(partition)

def test_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().test_partition(partition)

# you can find the mnist csv files here http://pjreddie.com/projects/mnist-in-csv/
def train_epochs(num_epochs, training_rdd):
	for i in range(num_epochs):
		mapped_training = training_rdd.mapPartitions(train_partition)
		accuracies = mapped_training.collect()
		print 'accuracies: %s' % accuracies
		training_rdd.repartition(training_rdd.getNumPartitions())

def test_all():
	testing_rdd = sc.textFile(directory + "molecular/molecular_test_all.csv")
	mapped_testing = testing_rdd.mapPartitions(test_partition)
	return mapped_testing.reduce(add)/mapped_testing.getNumPartitions()

def save_model():
	websock = websocket.create_connection('ws://localhost:55555')
	message = {'type': 'save_model'}
	websock.send(json.dumps(message))


def start_parameter_server(model, warmup_data):
	parameter_server = ParameterServer(model=model, warmup_data=warmup_data)
	parameter_server.start()
	return parameter_server


ex = Experiment('tensorspark')
ex.observers.append(MongoObserver.create(db_name='tensorspark_experiments'))
#model = mnistdnn.MnistDNN()
#model = higgsdnn.HiggsDNN()
model = moleculardnn.MolecularDNN()
conf = pyspark.SparkConf().set("spark.python.profile", "true")
sc = pyspark.SparkContext(conf=conf)

@ex.config
def configure_experiment():
	warmup_iterations = 10000
	num_epochs = 3

@ex.capture
@ex.automain
def main(warmup_iterations, num_epochs):
	try:
		training_rdd = sc.textFile(directory + "molecular/molecular_train_all.csv")
	#	training_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/medium_mnist_train.csv')
		warmup_data = training_rdd.take(warmup_iterations)
		parameter_server = start_parameter_server(model=model, warmup_data=warmup_data)
		#raw_input('Press enter to continue\n')

		training_rdd = training_rdd.subtract(sc.parallelize(warmup_data))
		train_epochs(num_epochs, training_rdd)
		print 'Done training'
		save_model()
		print 'Testing now'
		test_results = test_all()
		print test_results
		sc.show_profiles()
		return test_results
	finally:
		tornado.ioloop.IOLoop.current().stop()
