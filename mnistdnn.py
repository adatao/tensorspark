import tensorflow as tf
import numpy as np
from parameterservermodel import ParameterServerModel

def weight_variable(shape, name):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)

# 6 layers, 1024, 512 neurons

class MnistDNN(ParameterServerModel):
    def __init__(self, batch_size, gpu=True):
        #NUM_CORES = 4
	self.gpu = gpu	
        self.batch_size = batch_size
	session = tf.InteractiveSession()
#        session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	#session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES))
#        with session.graph.device(self.device_for_node):
	input_units = 784
	output_units = 10
	hidden_units = 1024
	x = tf.placeholder("float", shape=[None, input_units], name='x')
                #x_image = tf.reshape(x, [-1,28,28,1], name='reshape')
	true_y = tf.placeholder("float", shape=[None, output_units], name='y_')

	W_fc0 = weight_variable([input_units, hidden_units], 'W_fc0')
	b_fc0 = bias_variable([hidden_units], 'b_fc0')
	h_fc0 = tf.nn.relu(tf.matmul(x, W_fc0) + b_fc0)

	W_fc1 = weight_variable([hidden_units, hidden_units], 'W_fc1')
	b_fc1 = bias_variable([hidden_units], 'b_fc1')
	h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

	W_fc2 = weight_variable([hidden_units, output_units], 'W_fc2')
	b_fc2 = bias_variable([output_units], 'b_fc2')

	keep_prob = tf.Variable(0.5, name='keep_prob', trainable=False)
	h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

	guess_y = tf.matmul(h_fc1, W_fc2) + b_fc2
	guess_y_dropout = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

	variables = [W_fc0, b_fc0, W_fc1, b_fc1, W_fc2, b_fc2]
	loss = tf.nn.softmax_cross_entropy_with_logits(guess_y_dropout, true_y)

	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
	compute_gradients = optimizer.compute_gradients(loss, variables)
	apply_gradients = optimizer.apply_gradients(compute_gradients)
	minimize = optimizer.minimize(loss)
	correct_prediction = tf.equal(tf.argmax(guess_y,1), tf.argmax(true_y,1))
	error_rate = 1 - tf.reduce_mean(tf.cast(correct_prediction, "float"))

	ParameterServerModel.__init__(self, x, true_y, compute_gradients, apply_gradients, minimize, error_rate, session, batch_size)

    def process_data(self, data):
        batch_size = self.batch_size                                                                                                                                       
        #batch_size = 100                                                                                                                                                  
        num_classes = self.get_num_classes()                                                                                                                               
        features = []                                                                                                                                                      
        labels = []                                                                                                                                                        
        if batch_size == 0:                                                                                                                                                
            batch_size = len(data)                                                                                                                                         
        for line in data:                                                                                                                                                  
            #if len(line) is 0:                                                                                                                                             
            if len(line) < 2:  #mod (Fixing a dataset parsing error)                                                                                                                                          
                print 'Skipping empty line'                                                                                                                                
                continue                                                                                                                                                   
            label = [0] * num_classes                                                                                                                                      
            split = line.split(',')                                                                                                                                        
            split[0] = int(split[0])                                                                                                                                       
            if split[0] >= num_classes:                                                                                                                                    
                print 'Error label out of range: %d' % split[0]                                                                                                            
                continue                                                                                                                                                   
            features.append(split[1:])                                                                                                                                     
            label[split[0]] = 1                                                                                                                                            
            labels.append(label)                                                                                                                                           
                                                                                                                                                                           
        return labels, features                                                                                                                                            
                                                                                                                                                                           
    def process_partition(self, partition):                                                                                                                                
        batch_size = self.batch_size                                                                                                                                       
        #batch_size = 100                                                                                                                                                  
        print 'batch_size = %d' % batch_size                                                                                                                               
        num_classes = self.get_num_classes()                                                                                                                               
        features = []                                                                                                                                                      
        labels = []                                                                                                                                                        
        if batch_size == 0:                                                                                                                                                
            batch_size = 1000000                                                                                                                                           
        for i in xrange(batch_size):                                                                                                                                       
            try:                                                                                                                                                           
                line = partition.next()                                                                                                                                    
                if len(line) is 0:                                                                                                                                         
                    print 'Skipping empty line'                                                                                                                            
                    continue                                                                                                                                               
                label = [0] * num_classes                                                                                                                                  
                split = line.split(',')                                                                                                                                    
                split[0] = int(split[0])                                                                                                                                   
                if split[0] >= num_classes:                                                                                                                                
                    print 'Error label out of range: %d' % split[0]                                                                                                        
                    continue                                                                                                                                               
                features.append(split[1:])                                                                                                                                 
                label[split[0]] = 1                                                                                                                                        
                labels.append(label)                                                                                                                                       
            except StopIteration:                                                                                                                                          
                break                                                                                                                                                      
                                                                                                                                                                           
        return labels, features                                                                                                                                            
 
    def device_for_node(self, node):
	if node.type == 'MatMul' and self.gpu:
		return '/gpu:0'
	else:
		return '/cpu:0'


