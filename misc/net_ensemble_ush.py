import h2o
import pandas as pd
import numpy as np
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from sklearn import metrics
from ensemble_methods import plurality_ensemble#, avg_prob_ensemble
#change ip to hinton
use_ip = "192.168.1.90"
    
h2o.init(ip=use_ip, port="54321")
use_dir = "/home/ushnish/H2O_analysis/"
#Input data
#train = h2o.import_file(path= use_dir + "train_1_5.csv")
train = h2o.import_file(path= use_dir + "train_1_1_95_encoded_more_features.csv")
train_pd = pd.read_csv(use_dir + "train_1_1_95_encoded_more_features.csv")
predictors = list(train_pd.columns)[:-1]
test = h2o.import_file(path=use_dir + "test_no_label_1_1_95_encoded_more_features.csv")
test_df = pd.read_csv(use_dir + "test_1_1_95_encoded_more_features.csv")
nrows = train_pd.shape[0]
def getH2Obagged(random_features):
    train_bagged = train_pd.loc[np.random.choice(range(nrows), nrows / 2, replace = True)]
    train_bagged = train_bagged[random_features + [target]]
    train_bagged.to_csv(use_dir + "train_bagged.csv", encoding='ascii', index = False)
    train_bagged_h2o = h2o.import_file(path=local_dir + "train_bagged.csv")
    return train_bagged_h2o
from time import time
trials = 20
epochs = 3
num_models = 50
cv_folds = 10
num_bootstraps = 5

#Column for each  model, ensembles of 2 until n, ensembles of 2 until n
aucs = np.zeros([trials, 3*num_models - 2])
f1s = np.zeros([trials, 3*num_models - 2])
times = np.zeros([trials, num_models])
cv_mses = np.zeros([trials, num_models])

for i in xrange(trials):
    all_classifications = pd.DataFrame()
    all_probabilities = pd.DataFrame()    
    for j in xrange(num_models):    
        model1 = H2ODeepLearningEstimator(
                    stopping_rounds=1,
                    stopping_tolerance=0.0001,
#                    stopping_metric="misclassification",
                    hidden = hiddens,
                    loss = "Quadratic",
                    l1 = 0,
                    l2 = 0,
                    epochs = epochs,
                    activation = "Rectifier",
                    score_training_samples = 10000,
                    initial_weight_distribution = "UniformAdaptive",
                    single_node_mode = True,
                    adaptive_rate = True,
                    nfolds = cv_folds
            )
        num_features = int(len(predictors) * np.random.random()) + 1
        random_features = list(np.random.choice(predictors, num_features, replace = False))
        train_bagged = getH2Obagged(random_features)
        start = time()
        model1.train(random_features, target, training_frame = train_bagged)
        for k in xrange(num_bootstraps - 1):
            model = H2ODeepLearningEstimator(
                    checkpoint = model1,
                    stopping_rounds=1,
                    stopping_tolerance=0.0001,
#                    stopping_metric="misclassification",
                    hidden = hiddens,
                    loss = "Quadratic",
                    l1 = 0,
                    l2 = 0,
                    epochs = epochs,
                    activation = "Rectifier",
                    score_training_samples = 10000,
                    initial_weight_distribution = "UniformAdaptive",
                    single_node_mode = True,
                    adaptive_rate = True,
                    nfolds = cv_folds
            )
            train_bagged = getH2Obagged(random_features)
            model.train(random_features, target, training_frame = train_bagged)
            model1 = model
        end = time()
        times[i][j] = end - start                                 
        cv_mses[i][j] = model.mse(xval = True)                   
                                                                  
        predictions = model.predict(test)                       
        h2o.remove(model) 
        h2o.remove(model1)
        h2o.remove(train_bagged)                                         
        predictions = predictions.as_data_frame(use_pandas=True)['predict'].values
        classifications = np.where(predictions > 0.5, 1, 0)
        all_probabilities[j] = predictions 
        all_classifications[j] = classifications
        aucs[i][j] = metrics.roc_auc_score(test_df['risk'], predictions)
        print(aucs[i][j])
        f1s[i][j] = metrics.f1_score(test_df['risk'], classifications)
        print(f1s[i][j])   
        
    classifications_probabilities = pd.concat([all_classifications, all_probabilities], axis = 1)
    classifications_probabilities_matrix = classifications_probabilities.values  
    for ensemble_num in xrange(2, num_models + 1):
        j += 1
        plurality = plurality_ensemble(classifications_probabilities_matrix, ensemble_num)   
        aucs[i][j] = metrics.roc_auc_score(test_df['risk'], plurality, average = 'weighted')  
        f1s[i][j] = metrics.f1_score(test_df['risk'], plurality)  
    for ensemble_num in xrange(2, num_models + 1):    
        j += 1
        avg_prob = np.mean(classifications_probabilities_matrix[:,num_models:num_models + ensemble_num], axis = 1)
#        avg_prob = avg_prob_ensemble(classifications_probabilities_matrix, ensemble_num)    
        classifications = np.where(avg_prob > 0.5, 1, 0)
        aucs[i][j] = metrics.roc_auc_score(test_df['risk'], avg_prob) 
        f1s[i][j] = metrics.f1_score(test_df['risk'], classifications)        
    
    print(np.mean(aucs[0:i+1,:], axis = 0))
    print(np.mean(f1s[0:i+1,:], axis = 0))
    print(np.mean(times[0:i+1,:], axis = 0))
    print(np.mean(cv_mses[0:i+1,:], axis = 0))
