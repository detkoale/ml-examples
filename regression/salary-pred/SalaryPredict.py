
# coding: utf-8

# In[35]:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
import numpy as np


# In[36]:

train_data = pd.read_csv('salary-train.csv')
test_data = pd.read_csv('salary-test-mini.csv')


# In[37]:

print test_data


# In[38]:

#приведем все тексты к нижнему регистру
train_data['FullDescription'] = pd.Series(train_data['FullDescription']).str.lower()
#train_data['LocationNormalized'] = pd.Series(train_data['LocationNormalized']).str.lower()


# In[39]:

#меняем все, кроме букв и цифр, на пробелы
train_data['FullDescription'] = train_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


# In[69]:

#заменяем пропуски на слово nan
#print train_data.isnull().any()
train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)


# In[20]:

vectorizer = TfidfVectorizer(min_df=5)
X_train_descr = vectorizer.fit_transform(train_data['FullDescription'])
X_test_descr = vectorizer.transform(test_data['FullDescription'])


# In[63]:

X_test_descr.shape


# In[80]:

#one-hot кодирование категориальных признаков
enc = DictVectorizer()
X_train_loc = enc.fit_transform(train_data[['LocationNormalized']].to_dict('records'))
X_train_time = enc.transform(train_data[['ContractTime']].to_dict('records'))
#X_test_categ = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))


# In[81]:

#train_X = hstack([X_train_descr, X_train_categ])
train_X = hstack([X_train_descr, X_train_loc, X_train_time])


# In[82]:

print 'has nan', np.isnan(train_X.data).any()
print 'has inf', np.isinf(train_X.data).any()


# In[84]:

train_X.shape


# In[83]:

train_y = train_data['SalaryNormalized']
predictor = Ridge(alpha=1, random_state=241)
predictor.fit(train_X, train_y)


# In[88]:

X_test_loc = enc.transform(test_data[['LocationNormalized']].to_dict('records'))
X_test_time = enc.transform(test_data[['ContractTime']].to_dict('records'))
test_X = hstack([X_test_descr, X_test_loc, X_test_time])

test_y = predictor.predict(test_X)
print round(test_y[0],2)
print round(test_y[1],2)


# In[89]:

print test_y


# In[ ]:



