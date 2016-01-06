import tensorflow as tf
import numpy as np

rand1 = np.random.random((1000,1000))
rand2 = np.random.random((1000,1000))

matrix1 = tf.constant(rand1)
matrix2 = tf.constant(rand2)

import timeit
start = timeit.timeit()
tf.matmul(matrix1, matrix2)
end = timeit.timeit()
print end-start


a = np.array(rand1)
b = np.array(rand2)

start = timeit.timeit()
c = a.dot(b)
end = timeit.timeit()
print end-start
