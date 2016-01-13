import parameterwebsocketclient
import pyspark
from operator import add

def train_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().train_partition(partition)

def test_partition(partition):
	return parameterwebsocketclient.TensorSparkWorker().test_partition(partition)


# you can find the mnist csv files here http://pjreddie.com/projects/mnist-in-csv/
def train_all():
	training_rdd = sc.textFile('./mnist_train.csv')
	mapped_training = training_rdd.mapPartitions(train_partition)
	mapped_training.collect()

def test_all():
	testing_rdd = sc.textFile('./mnist_test.csv')
	mapped_testing = testing_rdd.mapPartitions(test_partition)
	return mapped_testing.reduce(add)/testing_rdd.count()

sc = pyspark.SparkContext()
train_all()
print test_all()