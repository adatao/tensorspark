import tensorflow as tf
from parameterservermodel import ParameterServerModel
import numpy as np
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.01)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(0.01, shape=shape)
   return tf.Variable(initial)
def xavier_init(shape):
  (fan_in, fan_out) = shape
  low = -1*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation 
  high = 1*np.sqrt(6.0/(fan_in + fan_out))
  return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))



class HiggsDNN(ParameterServerModel):
    def __init__(self, batch_size):

#        NUM_CORES = 4
#        session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES))

        num_hidden_units = 2048
        session = tf.InteractiveSession()
        input_units = 28
        output_units = 1
        x = tf.placeholder("float", shape=[None, input_units], name='x')
        true_y = tf.placeholder("float", shape=[None, output_units], name='y_')

#        W_fc1 = weight_variable([input_units, num_hidden_units])
        W_fc1 = weight_variable([input_units, num_hidden_units])
        b_fc1 = bias_variable([num_hidden_units])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

 #       W_fc2 = weight_variable([num_hidden_units, num_hidden_units])
        W_fc2 = weight_variable([num_hidden_units, num_hidden_units])
        b_fc2 = bias_variable([num_hidden_units])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  #      W_fc3 = weight_variable([num_hidden_units, num_hidden_units])
        W_fc3 = weight_variable([num_hidden_units, num_hidden_units])
        b_fc3 = bias_variable([num_hidden_units])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

   #     W_fc4 = weight_variable([num_hidden_units, output_units])
        W_fc4 = weight_variable([num_hidden_units, output_units])
        b_fc4 = bias_variable([output_units])

        keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)
        h_fc3_dropout = tf.nn.dropout(h_fc3, keep_prob)

        guess_y = tf.matmul(h_fc3, W_fc4) + b_fc4
        guess_y_dropout = tf.matmul(h_fc3_dropout, W_fc4) + b_fc4

        variables = [W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4]                                                                                               
       # loss =  tf.nn.l2_loss(guess_y_dropout - true_y)                                                                                                                   
        #loss = tf.reduce_mean(tf.square(tf.sign(guess_y_dropout) - tf.sign(true_y - 0.5)))                                                                                
        loss = tf.reduce_mean(tf.square(guess_y_dropout - true_y))                                                                                                         
#        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.99, beta2=0.999, epsilon=1e-06, use_locking=False, name='Adam')                                 
                                                                                                                                                                           
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.0000001, momentum=0.9, use_locking=False, name='momentum')                                                         
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.0000001, momentum=0.9995, use_locking=False, name='Momentum')                                               
       # optimizer = tf.train.RMSPropOptimizer(1e-4, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')                                            
        compute_gradients = optimizer.compute_gradients(loss, variables)                                                                                                   
        apply_gradients = optimizer.apply_gradients(compute_gradients)                                                                                                     
        minimize = optimizer.minimize(loss)                                                                                                                                
        #correct_prediction = tf.equal(tf.clip_by_value(tf.round(guess_y), 0.0, 1.0), true_y)                                                                              
        #correct_prediction = tf.equal(tf.round(guess_y), true_y)                                                                                                          
        #correct_prediction = tf.equal((tf.sign(guess_y) + 1)/2., true_y)                                                                                                  
        error_rate = loss                                                                                                                                                  
        #error_rate = 1 - tf.reduce_mean(tf.cast(correct_prediction, "float"))                                                                                             
#               correct_prediction = tf.equal(tf.argmax(guess_y,1), tf.argmax(true_y,1))                                                                                   
                                                                                                                                                                           
        ParameterServerModel.__init__(self, x, true_y, compute_gradients, apply_gradients, minimize, error_rate, session,batch_size)                                                        

                                                                                                                                                                           
                                                                                                                                                                           
    def process_data(self, data):                                                                                                                                          
        features = []                                                                                                                                                      
        batch_size = self.batch_size                                                                                                                                       
        labels = []                                                                                                                                                        
        #if batch_size == 0:                                                                                                                                               
        #    batch_size = len(data)                                                                                                                                        
        for line in data:                                                                                                                                                  
              if len(line) is 0:                                                                                                                                           
                 print 'Skipping empty line'                                                                                                                               
                 continue                                                                                                                                                  
              split = line.split(',')                                                                                                                                      
              features.append(split[:-1])                                                                                                                                  
              labels.append([float(split[-1])])                                                                                                                            
                                                                                                                                                                           
        return labels, features                                                                                                                                            
                                                                                                                                                                           
    def process_partition(self, partition):                                                                                                                                
        batch_size = self.batch_size                                                                                                                                       
        features = []                                                                                                                                                      
        labels = []                                                                                                                                                        
        #if batch_size == 0:                                                                                                                                               
        #    batch_size = 1000000                                                                                                                                          
        for i in xrange(batch_size):                                                                                                                                       
            try:                                                                                                                                                           
                line = partition.next()                                                                                                                                    
                if len(line) is 0:                                                                                                                                         
                    print 'Skipping empty line'                                                                                                                            
                    continue                                                                                                                                               
                split = line.split(',')                                                                                                                                    
                features.append(split[:-1])                                                                                                                                
                labels.append([float(split[-1])])                                                                                                                          
            except StopIteration:                                                                                                                                          
                break                                                                                                                                                      
        return labels, features                                                                                                                                            
