# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:36:14 2018

@author: m180234
"""

import numpy as np  #includes math eqns
import matplotlib.pyplot as plt  #use to plot data
import pandas as pd  #import and manage datasets


#import the dataset now
# '[:, number]' means we take all the indices but then only read the index specified
dataset = pd.read_csv('breast_cancer_wisconsin.csv', na_values=['?']) # turns ? into nan
X = dataset.iloc[:, 1:10].values  # independent variables
y = dataset.iloc[:, 10].values  # dependent variables


# take care of missing data by inputting the mean of the column in the blank
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])


# Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# machine interprets training set and uses what it learns to predict dep.var in test set


# LOGISTIC REGRESSION NEEDS FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


"""
Pre-processing is done, now time to classify: Since this is medical and false
negatives could be extremely bad, I will use logreg so that I can see probabilistic
classification.
"""

# Fitting Log Regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# predict test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix # this is a function not a class
cm = confusion_matrix(y_test, y_pred)
# this tells us how many predictions we got right and wrong


# this gives the probabilities of the sample being malignant/benign
probability = classifier.predict_proba(X_test)[:,1]























