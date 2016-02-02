# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:26:14 2016

@author: ushnishde
"""

import tensorflow as tf
from time import time
import numpy as np
from sklearn import metrics
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)
def xavier_init(shape):
  (fan_in, fan_out) = shape
  low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation 
  high = 4*np.sqrt(6.0/(fan_in + fan_out))
  return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)

def model3(x, num_hidden_units, keep_prob):
  keep_prob = tf.constant(keep_prob)
  W_fc1 = weight_variable([28, num_hidden_units])
  b_fc1 = bias_variable([num_hidden_units])
  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([num_hidden_units, num_hidden_units])
  b_fc2 = bias_variable([num_hidden_units])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  W_fc3 = weight_variable([num_hidden_units, num_hidden_units])
  b_fc3 = bias_variable([num_hidden_units])
  h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
  h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)


  W_fc4 = weight_variable([num_hidden_units, 1])
  b_fc4 = bias_variable([1])
  return tf.matmul(h_fc3_drop, W_fc4) + b_fc4

x = tf.placeholder(tf.float32, [None, 28], name = "x")
y = tf.placeholder(tf.float32, shape=[None, 1], name = "y")
y_pred = model3(x, 1024, 0.5)
quad_loss = tf.nn.l2_loss(y_pred - y)
train_op = tf.train.AdagradOptimizer(1e-3).minimize(quad_loss)

local_dir = "/home/ushnish/TensorSpark/higgs/"
higgs_train = np.load(local_dir + 'higgs_train.npy')
higgs_train_labels = np.load(local_dir + 'higgs_train_labels.npy')
higgs_val = np.load(local_dir + 'higgs_val.npy')
higgs_val_labels = np.load(local_dir + 'higgs_val_labels.npy')
higgs_test = np.load(local_dir + 'higgs_test.npy')
higgs_test_labels = np.load(local_dir + 'higgs_test_labels.npy')
train_rows = higgs_train.shape[0]

sess = tf.Session()
sess.run(tf.initialize_all_variables())
MINI_BATCH_SIZE = 50
NUM_EPOCHS = 3 
print("Started training...")                                                                                                                                                   
starttime = time()                                                                                                                                                             
for i in range(NUM_EPOCHS):                                                                                                                                                    
  print("Num epochs = %d" % (i+1))                                                                                                                                             
  for start in range(0, train_rows - MINI_BATCH_SIZE, MINI_BATCH_SIZE):                                                                                                        
    end = start + MINI_BATCH_SIZE                                                                                                                                              
#    print(start, end)                                                                                                                                                         
    #print(higgs_train[start : end, :].shape, higgs_train_labels[start : end].shape)                                                                                           
    sess.run(train_op, feed_dict = {x : higgs_train[start : end, :], y : higgs_train_labels[start : end]})                                                                     
    if start % 50000 == 0:                                                                                                                                                     
       sample_rows = np.random.choice(higgs_val.shape[0], 20000, replace = False)                                                                                              
       val_mse = sess.run(quad_loss, feed_dict = {x: higgs_val[sample_rows, :], y: higgs_val_labels[sample_rows]})                                                             
       print("Step %d, MSE %g" % (start, val_mse))                                                                                                                             
endtime = time()                                                                                                                                                               
y_pred_final = np.zeros(higgs_test_labels.shape)                                                                                                                               
                                                                                                                                                                               
batch_size = 1000                                                                                                                                                              
for start in range(0, train_rows - batch_size, batch_size):                                                                                                                    
  end = start + batch_size                                                                                                                                                     
  y_pred_final[start : end] = sess.run(y_pred, feed_dict = {x: higgs_test[start : end, :]})                                                                                    
print("Total time taken = " + str(endtime - starttime))                                                                                                                        
print("Test MSE = %g" % (np.sum((y_pred_final - higgs_test_labels)**2)))                                                                                                       
print("AUC = %g" % (metrics.roc_auc_score(higgs_test_labels, y_pred_final, average="weighted")))                                                                               
predicted_labels = np.where(y_pred_final < 0.5, 0, 1)                                                                                                                          
print("F1 Score = %g" % (metrics.f1_score(higgs_test_labels, predicted_labels, average="weighted")))                                                                           