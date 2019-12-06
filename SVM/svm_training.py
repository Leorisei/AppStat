# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:10:13 2019

@author: 80686
"""

import pandas as pd
import os, fnmatch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats
# Random Seed
from numpy.random import seed

test_size_ratio = 0.10

if __name__ == "__main__":
    seed(4)

    # read the data
    df = pd.read_csv("train_test_2019.csv")

    df.columns = ['age','workclass','fnlwgt','education', 'education-num', 'marital-status', 'occupation','relationship', 'race','sex','capital-gain','capital-loss','hours-per-week','native-country','y']
    all_classes = df['y']
    classes=['yes','no']
    color_dict={'yes':'blue', 'no':'red'}
    
    all_inputs = df[['age','workclass','fnlwgt','education', 'education-num', 'marital-status', 'occupation','relationship', 'race','sex','capital-gain','capital-loss','hours-per-week','native-country']]
    #  transform the str data to number 
    all_inputs[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]=all_inputs[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']].apply(LabelEncoder().fit_transform)
    all_inputs = np.array(all_inputs)
    
    max_values=np.max(all_inputs,axis=0)
    min_values=np.min(all_inputs,axis=0)
    ranges=max_values-min_values
    
#    
##  standarize the values and map the values from 0 to 1000
    for i in range(all_inputs.shape[1]):
        mean=all_inputs[:,i].mean()
        std=all_inputs[:,i].std()
        all_inputs[:,i]=(all_inputs[:,i]-mean)*1000/std

#    
    # Encode Labels and transform the label to snumber
    labelencoder = LabelEncoder()
    labelencoder.fit(all_classes)
    labelencoder.classes_
    classes_num = labelencoder.transform(all_classes)
    #train the data with SVM
    param_grid ={'C':[0.001,0.01,0.1,1e1,1e2],#C
                 'gamma':[0.0001,0.0005,0.001,0.01,0.1,0.5],}#gamma

#   Create Train and Test Set
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size_ratio, random_state=None)
    splits = splitter.split(all_inputs, classes_num)
    for train_index, test_index in splits:
        train_set = all_inputs[train_index]
        test_set = all_inputs[test_index]
        train_classes = classes_num[train_index]
        test_classes = classes_num[test_index]
    print(train_set[1],train_set[2])
    svclassifier = GridSearchCV(SVC(kernel='rbf'),param_grid)
    svclassifier=svclassifier.fit(all_inputs,classes_num)
    
    
    print("Best: %f using %s" % (svclassifier.best_score_,svclassifier.best_params_))
#    
    print("Best estimotor found by grid search:")
    predicted_labels = svclassifier.predict(test_set)
    print(predicted_labels)
    # Recall - the ability of the classifier to find all the positive samples  TP/(TP+FN)
    print("Recall: ", recall_score(test_classes, predicted_labels,average=None))
    
    # Precision - The precision is intuitively the ability of the classifier not to 
    #label as positive a sample that is negative          TP/(TP+FP)
    print("Precision: ", precision_score(test_classes, predicted_labels,average=None))
    
    # F1-Score - The F1 score can be interpreted as a weighted average of the precision 
    #and recall
    print("F1-Score: ", f1_score(test_classes, predicted_labels, average=None))
    
    # Accuracy - the number of correctly classified samples
    print("Accuracy: %.2f  ," % accuracy_score(test_classes, predicted_labels,normalize=True))
    print("Number of samples:",test_classes.shape[0])
    # Save the model
    joblib.dump(svclassifier, 'svclassifier2.joblib')