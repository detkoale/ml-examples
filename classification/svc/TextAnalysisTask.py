
# coding: utf-8

# In[165]:

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import numpy as np


# In[81]:

#Загружаем тестовую выборку новостей
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )


# In[82]:

#Преобразовываем тексты TF-IDF
textTransformator = TfidfVectorizer()
X = textTransformator.fit_transform(newsgroups.data)
y = newsgroups.target


# In[83]:

#Совершаем перебор параметров, чтобы найти оптимальный набор
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)


# In[86]:

#print max(a.mean_validation_score for a in gs.grid_scores_)
#print gs.best_estimator_
#print gs.best_params_


# In[87]:

#for a in gs.grid_scores_:
#    print a.mean_validation_score, a.parameters
    


# In[85]:

#Обучаем классификатор с оптимальными параметрами, найденными на предыдущем шаге 
clf = gs.best_estimator_
clf.fit(X,y)


# In[166]:

feature_mapping = textTransformator.get_feature_names()
d = dict(zip(clf.coef_.indices, clf.coef_.data))
sorted_d = sorted(d.items(), key = lambda x: abs(x[1]), reverse=True)
#for key, value in sorted_d:
#    print feature_mapping[key], value
#print clf.coef_.indices[1], clf.coef_.data[1], feature_mapping[clf.coef_.indices[1]]
response = [str(feature_mapping[key]) for key,value in sorted_d[:10]]
print response


# In[169]:

clf.classes_


# In[ ]:



