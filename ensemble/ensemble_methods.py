# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:20:23 2015

@author: ushnishde
"""

# both methods return an array of size (nrows, 1) each element is result of ensemble on a training row

import numpy as np
import scipy.stats
# assuming array of the form prediction repeated n times, then p1 repeated n times
# num is 2 to n
# with even num have to look at average prob of being 1 and predict 1 if >= 0.5
def plurality_ensemble(matrix, num):
    rows = matrix.shape[0]
    num_models = matrix.shape[1] / 2
    if num > num_models:
        print("Wrong dimensions")
        return
    result = np.zeros(rows)
    for i in xrange(rows):
        array = matrix[i, :]
        bin_counts = np.bincount(array[0:num].astype(np.int64))
        # assuming 2 classes, if the total number of models voting for a class is more than half, take it
        if num % 2 != 0 or 2 * max(bin_counts) > num:
            result[i] = np.argmax(bin_counts)
        # if exactly half the models vote for a class, take average of that many probabilities
        else:
            avg_prob = np.mean(array[num_models:num_models + num])
            if avg_prob >= 0.5:
                result[i] = 1
            else:
                result[i] = 0
    return result

# just take average of required number of elements in each row and output 1 if >= 0.5
def avg_prob_ensemble(matrix, num):
    rows = matrix.shape[0]
    num_models = matrix.shape[1] / 2
    result = np.zeros(rows)
    for i in xrange(rows):
        array = matrix[i, :]
        if np.mean(array[num_models:num_models + num]) >= 0.5:
            result[i] = 1
        else:
            result[i] = 0
    return result

# plurality from multi classes
def plurality_ensemble_multi(matrix, num_ensembles):
    rows = matrix.shape[0]
    result = np.zeros(rows)
    for i in xrange(rows):
        array = matrix[i, :]
        bin_counts = np.bincount(array[0:num_ensembles].astype(np.int64))
        result[i] = np.argmax(bin_counts)
    return result

def probability_ensemble(matrix, num_classes, num_ensembles, avg_method):
    rows = matrix.shape[0]
    class_probs = np.zeros([rows, num_classes])
    for j in xrange(num_classes):
#        [0, 3, 6], [1, 4, 7] etc
        indices = np.array(range(0, num_ensembles)) * num_classes + j
        for i in xrange(rows):
            array = matrix[i, :]
            probs = array[indices]
            if avg_method == 'mean':
                class_probs[i][j] = np.mean(probs)
            elif avg_method == 'hmean':
                class_probs[i][j] = scipy.stats.hmean(probs)
            elif avg_method == 'gmean':
                class_probs[i][j] = scipy.stats.gmean(probs)
            else:
                print("Use mean, hmean or gmean!!")
                return
    results = np.argmax(class_probs, axis = 1)
    return results
        