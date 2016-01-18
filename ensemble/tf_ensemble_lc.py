# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:33:29 2016

@author: ushnishde
"""

from sklearn import cross_validation
import pandas as pd
import numpy as np
import tensorflow as tf
#from transform_data import clean_features
import time
import skflow, sys
from sklearn import preprocessing, metrics

local_dir = '/Users/ushnishde/Documents/TensorSpark/'
output_file = "tf_ensembles_output.txt"
with open(output_file, 'w') as f:
    f.write('')

target = 'risk'
all_predictors = ['dti', 'delinq_2yrs', 'pub_rec', 'total_acc', 'purpose', 'zip_code', 'addr_state', 'inq_last_6mths', 'emp_length', 'verification_status', 'open_acc', 'revol_util', 'loan_amnt', 'home_ownership', 'fico_range_low','int_rate_num', 'issue_d_time', 'grade']

training_data = pd.read_csv(local_dir + "train_final_lc.csv")
training_labels = training_data[target]
validation_data = pd.read_csv(local_dir + "test_final_lc.csv")
validation_labels = validation_data[target]

nrows = training_data.shape[0]

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, max_depth = 20).fit(training_data[all_predictors], training_labels.values.ravel())
   #print(test_final_no_label.columns)    
   #print(train_final_no_label.columns)
rfc_predict = rfc.predict(validation_data[all_predictors])
print(metrics.f1_score(validation_labels, rfc_predict, average='weighted'))

predictions = rfc.predict_proba(validation_data[all_predictors])
print(metrics.roc_auc_score(validation_labels, predictions[:,1], average = 'weighted'))

with open(output_file, 'a') as f:
    f.write("Random Forest Output\n")
    f.write("f1 = " + str(metrics.f1_score(validation_labels, rfc_predict, average='weighted')) + "\n")
    f.write("auc = " + str(metrics.roc_auc_score(validation_labels, predictions[:,1], average = 'weighted')) + "\n")

from ensemble_methods import plurality_ensemble_multi, probability_ensemble
scaler = preprocessing.StandardScaler()
scaler.fit(training_data[all_predictors].append(validation_data[all_predictors]))
train_scaled_values = scaler.transform(training_data[all_predictors])
training_values_scaled = pd.DataFrame(train_scaled_values, columns = [all_predictors])
validation_values_scaled = scaler.transform(validation_data[all_predictors])
validation_values_scaled = pd.DataFrame(validation_values_scaled, columns = [all_predictors])

hidden_layers = [100]

num_classes = 2                                                                                                                                                        
                                                                                                                                                                       
dnnClassifier = skflow.TensorFlowDNNClassifier(                                                                                                                        
                continue_training = True,                                                                                                                              
                hidden_units=hidden_layers,                                                                                                                            
                n_classes=1,                                                                                                                                 
                batch_size=228,                                                                                                                                        
                steps=7000,                                                                                                                                            
                learning_rate=0.15,                                                                                                                                    
                verbose = 0)                                                                                                                                           
                                                                                                                                                                       
dnnClassifier.fit(training_values_scaled, training_labels)                                                                                                             
pred_prob = dnnClassifier.predict_proba(validation_values_scaled)                                                                                                      
predictions = np.argmax(pred_prob, axis = 1)                                                                                                                           
with open(output_file, 'a') as f:                                                                                                                                      
    f.write("1 NN Output\n")                                                                                                                                           
    f.write("f1 = " + str(metrics.f1_score(validation_labels, predictions, average='weighted')) + "\n")                                                                
    f.write("auc = " + str(metrics.roc_auc_score(validation_labels, pred_prob[:,1], average='weighted')))                                                              
print(metrics.f1_score(validation_labels, predictions, average='weighted'))                                                                                            
print(metrics.roc_auc_score(validation_labels, pred_prob[:,1], average='weighted'))                                                                                    
                                                                                                                                                 
num_models = 10                                                                                                                                                        
num_bootstraps = 5                                                                                                                                                 
                                                                                                                                                                       
trials = 5                                                                                                                                                             
f1s = np.zeros([trials, 5*num_models - 4])                                                                                                                             
times = np.zeros([trials, num_models])                                                                                                                                 
nrows = training_values_scaled.shape[0]                                                                                                                                
                                                                                                                                                                       
def getBagged(random_features):                                                                                                                                                       
    bagged_rows = np.random.choice(range(nrows), nrows / num_bootstraps, replace = True)                                                                              
    train_bagged = training_values_scaled.iloc[bagged_rows]                                                                                                            
    train_bagged_labels = training_labels.iloc[bagged_rows]                                                                                                            
    return train_bagged[random_features], train_bagged_labels                                                                                                                           

for i in xrange(trials):
    all_probability_matrices = []                                                                                                                                      
    all_prediction_vectors = []                                                                                                                                        
    for j in xrange(num_models):                                                                                                                                       
        dnnClassifier = skflow.TensorFlowDNNClassifier(                                                                                                                
                continue_training = True,                                                                                                                              
                hidden_units=hidden_layers,                                                                                                                            
                n_classes=num_classes,                                                                                                                                 
                batch_size=228,                                                                                                                                        
                steps=7000,                                                                                                                                            
                learning_rate=0.15,                                                                                                                                    
                verbose = 0)                                                                                                                                           
        num_features = int(len(all_predictors) * np.random.random()) + 1                                                                                              
        random_features = list(np.random.choice(all_predictors, num_features, replace = False))                                                                       
        start = time.time()                                                                                                                                            
        for k in xrange(num_bootstraps):                                                                                                                               
            train_bagged, train_bagged_labels = getBagged(random_features)                                                                                                            
            dnnClassifier.fit(train_bagged, train_bagged_labels)                                                                                                       
        end = time.time()                                                                                                                                              
        print(str(j) + " - " + time.strftime('%X %x %Z'))                                                                                                              
        pred_prob = dnnClassifier.predict_proba(validation_values_scaled[random_features])                                                                                              
        all_probability_matrices.append(pred_prob)                                                                                                                     
        predictions = np.argmax(pred_prob, axis = 1)                                                                                                                   
        all_prediction_vectors.append(np.reshape(predictions, (predictions.shape[0], 1)))                                                                              
        f1s[i][j] = metrics.f1_score(validation_labels.values, predictions, average= 'weighted')                                                                       
        print(f1s[i][j])                                                                                                                                               
        times[i][j] = end - start                                                                                                                                      
                                                                                                                                                                       
    all_probabilities = np.concatenate(all_probability_matrices, axis = 1)                                                                                             
    all_predictions = np.concatenate(all_prediction_vectors, axis = 1)                                                                                                 
                                                                                                                                                                       
    for ensemble_num in xrange(2, num_models + 1):                                                                                                                     
        j += 1                                                                                                                                                         
        plurality_result = plurality_ensemble_multi(all_predictions, ensemble_num)                                                                                     
        f1s[i][j] = metrics.f1_score(validation_labels.values, plurality_result, average= 'weighted')                                                                  
    for ensemble_num in xrange(2, num_models + 1):                                                                                                                     
        j += 1                                                                                                                                                         
        mean_result = probability_ensemble(all_probabilities, num_classes, ensemble_num, 'mean')                                                                       
        f1s[i][j] = metrics.f1_score(validation_labels.values, mean_result, average= 'weighted')                                                                       
                                                                                                                                                                       
    all_probabilities_pos = np.where(all_probabilities <= 10e-45, 10e-45, all_probabilities)                                                                           
                                                                                                                                                                       
    for ensemble_num in xrange(2, num_models + 1):                                                                                                                     
        j += 1                                                                                                                                                         
        gmean_result = probability_ensemble(all_probabilities_pos, num_classes, ensemble_num, 'gmean')                                                                 
        f1s[i][j] = metrics.f1_score(validation_labels.values, gmean_result, average= 'weighted')                                                                      
    for ensemble_num in xrange(2, num_models + 1):                                                                                                                     
        j += 1                                                                                                                                                         
        hmean_result = probability_ensemble(all_probabilities_pos, num_classes, ensemble_num, 'hmean')                                                                 
        f1s[i][j] = metrics.f1_score(validation_labels.values, hmean_result, average= 'weighted')                                                                      
    f1_means = np.mean(f1s[0:i+1,:], axis = 0)                                                                                                                                       
    time_means = np.mean(times[0:i+1,:], axis = 0)                                                                                                                     
    with open(output_file, 'a') as f:                                                                                                                                                
        f.write("Individual\n")                                                                                                                                        
        f.write("f1s = " + str(f1_means[0 : num_models]) + "\n")                                                                                                                     
        f.write("times = " + str(time_means) + "\n")                                                                                                                                 

        f.write("Plurality\n")
        f.write("f1s = " + str(f1_means[num_models : 2*num_models - 1]) + "\n")

        f.write("Average Prob\n")
        f.write("f1s = " + str(f1_means[2*num_models - 1 : 3*num_models - 2]) + "\n")

        f.write("Geometric Avg Prob\n")
        f.write("f1s = " + str(f1_means[3*num_models - 2 : 4*num_models - 3]) + "\n")

        f.write("Harmonic Avg Prob\n")
        f.write("f1s = " + str(f1_means[4*num_models - 3 : 5*num_models - 4]) + "\n\n")
                                                                                                                                                                                     
                                                                          