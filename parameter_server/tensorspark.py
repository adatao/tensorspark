import parameterwebsocketclient
import pyspark
from operator import add
import websocket
import json

def train_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().train_partition(partition)

def test_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().test_partition(partition)

# you can find the mnist csv files here http://pjreddie.com/projects/mnist-in-csv/
def train_epochs(num_epochs, batch_size):
#	training_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/mnist_train.csv')
	training_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/medium_mnist_train.csv')
	for i in range(num_epochs):
		training_rdd.repartition(training_rdd.count()/batch_size)
		mapped_training = training_rdd.mapPartitions(train_partition)
		mapped_training.collect()

def test_all():
	testing_rdd = sc.textFile('/Users/christophersmith/code/adatao/tensorspark/data/mnist_test.csv')
	mapped_testing = testing_rdd.mapPartitions(test_partition)
	return mapped_testing.reduce(add)/mapped_testing.getNumPartitions()

def save_model():
	message = {'type': 'save_model'}
	websock.send(json.dumps(message))

sc = pyspark.SparkContext()
websock = websocket.create_connection('ws://localhost:55555')
num_epochs = 3
batch_size = 10
train_epochs(num_epochs, batch_size)
save_model()
print test_all()
