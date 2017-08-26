#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:47:52 2017

@author: Haoran Zhang
"""

import pandas as pd
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold
from sklearn.metrics  import confusion_matrix

df = pd.read_csv('creditcard.csv', header=0)

print(df.info())
print('====================================================================')
print(df.describe())
print('====================================================================')
classes = df['Class'].value_counts(normalize=True)
print(classes)
print("Percentage of Fraud transacation is "+str(classes[1]*100)+'%')

y = df['Class'].copy()
X = df.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


def classification(model, X_train, y_train, X_test, y_test):
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = confusion_matrix(y_test, y_pred)
    return metrics


model_set = [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier(), SVC(),]
for model in model_set:
    me = classification(model, X_train, y_train, X_test, y_test)
    print(str(model))
    print(me)