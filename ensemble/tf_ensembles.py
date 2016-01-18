# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:59:47 2016

@author: ushnishde
"""
from sklearn import cross_validation
import pandas as pd
import numpy as np
import tensorflow as tf
from transform_data import clean_features
from time import time
import skflow, sys
from sklearn import preprocessing, metrics

local_dir = '/Users/ushnishde/Documents/TensorSpark/'
training_data = pd.read_csv(local_dir + "training_data_numeric.csv")
training_labels = pd.read_csv(local_dir + "training_labels_numeric.csv")
nrows = training_data.shape[0]
validation_rows = np.random.choice(range(nrows), nrows / 10, replace = False) 
training_rows = np.setdiff1d(range(nrows), validation_rows)
validation_values = training_data.iloc[validation_rows]
validation_labels = training_labels.iloc[validation_rows]
training_values_final = training_data.iloc[training_rows]
training_labels_final = training_labels.iloc[training_rows]
predictors = list(training_data.columns)

#%%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=30, criterion='gini', max_features='auto',
                             max_depth=19, bootstrap=True, oob_score = True, n_jobs = -1)
rfc.fit(training_values_final.values, training_labels_final['status_group'].values.ravel()) 
#print(metrics.roc_auc_score(validation_labels['status_group'].values, rfc.predict_proba(validation_values.values), average= 'weighted'))                            
print(metrics.f1_score(validation_labels['status_group'].values, rfc.predict(validation_values.values), average= 'weighted'))                            
#%%
from ensemble_methods import plurality_ensemble_multi, probability_ensemble
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
training_values_scaled = scaler.transform(training_values_final)  
training_values_scaled = pd.DataFrame(training_values_scaled, columns = training_data.columns)
validation_values_scaled = scaler.transform(validation_values)                             
validation_values_scaled = pd.DataFrame(validation_values_scaled, columns = training_data.columns)
                             
num_classes = 3

num_models = 30
num_bootstraps = 10

trials = 2
#Column for each  model, ensembles of 2 until n, ensembles of 2 until n
#aucs = np.zeros([trials, 5*num_models - 4])
f1s = np.zeros([trials, 5*num_models - 4])
times = np.zeros([trials, num_models])
nrows = training_values_scaled.shape[0]

def getBagged(random_features):
    bagged_rows = np.random.choice(range(nrows), nrows / num_bootstraps, replace = True)
    train_bagged = training_values_scaled.iloc[bagged_rows]
    train_bagged_labels = training_labels_final.iloc[bagged_rows]
    return train_bagged[random_features], train_bagged_labels
    
for i in xrange(trials):
    all_probability_matrices = []  
    all_prediction_vectors = []
    for j in xrange(num_models):  
        dnnClassifier = skflow.TensorFlowDNNClassifier(
                continue_training = True,
                hidden_units=[1000,1000,1000], 
                n_classes=num_classes, 
                batch_size=228, 
                steps=4000, 
                learning_rate=0.15,
                verbose = 0) 
                
        num_features = int(len(predictors) * np.random.random()) + 1
        random_features = list(np.random.choice(predictors, num_features, replace = False))
        start = time()
        for k in xrange(num_bootstraps):
            train_bagged, train_bagged_labels = getBagged(random_features)
            dnnClassifier.fit(train_bagged, train_bagged_labels['status_group'])
        end = time()
        pred_prob = dnnClassifier.predict_proba(validation_values_scaled[random_features])
        all_probability_matrices.append(pred_prob)
        predictions = np.argmax(pred_prob, axis = 1)
        all_prediction_vectors.append(np.reshape(predictions, (predictions.shape[0], 1)))
        f1s[i][j] = metrics.f1_score(validation_labels['status_group'].values, predictions, average= 'weighted')
        times[i][j] = end - start
    
    all_probabilities = np.concatenate(all_probability_matrices, axis = 1)
    all_predictions = np.concatenate(all_prediction_vectors, axis = 1)
    
    for ensemble_num in xrange(2, num_models + 1):
        j += 1
        plurality_result = plurality_ensemble_multi(all_predictions, ensemble_num)
        f1s[i][j] = metrics.f1_score(validation_labels['status_group'].values, plurality_result, average= 'weighted')
    for ensemble_num in xrange(2, num_models + 1):
        j += 1
        mean_result = probability_ensemble(all_probabilities, num_classes, ensemble_num, 'mean')
        f1s[i][j] = metrics.f1_score(validation_labels['status_group'].values, mean_result, average= 'weighted')
        
    all_probabilities_pos = np.where(all_probabilities <= 10e-45, 10e-45, all_probabilities)
    
    for ensemble_num in xrange(2, num_models + 1):
        j += 1
        gmean_result = probability_ensemble(all_probabilities_pos, num_classes, ensemble_num, 'gmean')
        f1s[i][j] = metrics.f1_score(validation_labels['status_group'].values, gmean_result, average= 'weighted')
    for ensemble_num in xrange(2, num_models + 1):
        j += 1
        hmean_result = probability_ensemble(all_probabilities_pos, num_classes, ensemble_num, 'hmean')
        f1s[i][j] = metrics.f1_score(validation_labels['status_group'].values, hmean_result, average= 'weighted')
    
    print(np.mean(f1s[0:i+1,:], axis = 0))
    print(np.mean(times[0:i+1,:], axis = 0))