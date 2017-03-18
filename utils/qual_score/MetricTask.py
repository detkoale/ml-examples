
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

data = pd.read_csv("classification.csv")
print(data)


# In[3]:

cnf_matrix = confusion_matrix(data['true'], data['pred'])
TN = cnf_matrix[0][0]
FN = cnf_matrix[1][0]
TP = cnf_matrix[1][1]
FP = cnf_matrix[0][1]


# In[7]:

Accuracy = accuracy_score(data['true'], data['pred'])
Precision = precision_score(data['true'], data['pred'])
Recall = recall_score(data['true'], data['pred'])
F_metric = f1_score(data['true'], data['pred'])
print round(Accuracy,2), round(Precision,2), round(Recall, 2), round(F_metric, 2)


# In[25]:

scores_data = pd.read_csv('scores.csv')
#print scores_data


# In[26]:

logreg_score = roc_auc_score(scores_data['true'], scores_data['score_logreg'])
svm_score = roc_auc_score(scores_data['true'], scores_data['score_svm'])
knn_score = roc_auc_score(scores_data['true'], scores_data['score_knn'])
tree_score = roc_auc_score(scores_data['true'], scores_data['score_tree'])
print round(logreg_score, 2), round(svm_score, 2), round(knn_score,2), round(tree_score, 2)


# In[27]:

logreg_precision = precision_recall_curve(scores_data['true'], scores_data['score_logreg'])
#print logreg_precision
#print len(logreg_precision[0])


# In[28]:

max_logreg_prec_score = max(logreg_precision[0][i] for i in np.arange(0,len(logreg_precision[0])) if logreg_precision[1][i] >= 0.7)
print max_logreg_prec_score


# In[29]:

svm_prec = precision_recall_curve(scores_data['true'], scores_data['score_svm'])
max_svm_prec_score = max(svm_prec[0][i] for i in np.arange(0,len(svm_prec[0])) if svm_prec[1][i] >= 0.7)
print max_svm_prec_score


# In[30]:

knn_prec = precision_recall_curve(scores_data['true'], scores_data['score_knn'])
max_knn_prec_score = max(knn_prec[0][i] for i in np.arange(0,len(knn_prec[0])) if knn_prec[1][i] >= 0.7)
print max_knn_prec_score


# In[31]:

tree_curve = precision_recall_curve(scores_data['true'], scores_data['score_tree'])
max_tree_prec_score = max(tree_curve[0][i] for i in np.arange(0,len(tree_curve[0])) if tree_curve[1][i] >= 0.7)
print max_tree_prec_score


# In[ ]:



