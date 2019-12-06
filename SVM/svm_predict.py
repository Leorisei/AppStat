# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:59:39 2019

@author: 80686
"""
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
if __name__ == "__main__":
    
    
    #read the predicted data
    df = pd.read_csv("submit_2019.csv")
    df.columns = ['age','workclass','fnlwgt','education', 'education-num', 'marital-status', 'occupation','relationship', 'race','sex','capital-gain','capital-loss','hours-per-week','native-country','y']
    
    all_inputs = df[['age','workclass','fnlwgt','education', 'education-num', 'marital-status', 'occupation','relationship', 'race','sex','capital-gain','capital-loss','hours-per-week','native-country']]
    all_inputs[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]=all_inputs[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']].apply(LabelEncoder().fit_transform)
    all_inputs = np.array(all_inputs)
    #load the trained model

    max_values=np.max(all_inputs,axis=0)
    min_values=np.min(all_inputs,axis=0)
    ranges=max_values-min_values
    
#   standarize the values and map the values from 0 to 1000
    for i in range(all_inputs.shape[1]):
#        all_inputs[:,i] = (all_inputs[:,i]-min_values[i])*1000/ranges[i]
        mean=all_inputs[:,i].mean()
        std=all_inputs[:,i].std()
        all_inputs[:,i]=(all_inputs[:,i]-mean)*1000/std
        
        
    svclassifier = joblib.load('svclassifier2.joblib') 
    predicted_label=svclassifier.predict(all_inputs)
    print(predicted_label)
#    print(np.sum(predicted_label == '1'))
    np.savetxt('predicted_svm_result.txt',predicted_label,fmt="%.f")