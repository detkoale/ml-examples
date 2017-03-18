
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy as np



data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

#columns from 0 to len-1 contains features 
X = data[data.columns[:-1]]
#last column contains target 
y = data[data.columns[-1:]]

crossvalidation = KFold(random_state=1, n_splits=5, shuffle=True)
#for train_index, test_index in crossvalidation.split(X):
#    print ("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

def r2_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return r2_score(y, predictions)

#train randon forest model
for i in np.arange(0,50):
    print "Train random forest with estimator=",i+1
    clf = RandomForestRegressor(n_estimators=i+1, random_state=1)
    scorer = cross_val_score(clf, X = X, y = y, scoring = r2_scorer, 
                             cv = crossvalidation)    
#        clf.fit(X.iloc[train_index], y.iloc[train_index])
#            score = r2_score(y.iloc[test_index], clf.predict(X.iloc[test_index]))
#            scores.append(score)  
    print "R2 metric =", scorer, "Mean scorer", np.mean(scorer)


# In[ ]:



