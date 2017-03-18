
# coding: utf-8

# In[55]:

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# In[75]:

#загружаем цены на каждый день закрытия
price_data = pd.read_csv('close_prices.csv')
print price_data


# In[38]:

#price_data['date'].apply(str)
#price_data['date'] = price_data['date'].astype(str)
#print price_data['date']

#X = price_data.values
print price_data.iloc[:,1:]


# In[77]:

#обучаем principal component analysis с числом компонент = 10
analysis = PCA(n_components=10)
analysis.fit(price_data.iloc[:,1:])


# In[78]:

print analysis.explained_variance_ratio_


# In[79]:

#сколько компонент нужно, чтобы объяснить 90% дисперсии
n = 0
sumi = 0
for i in analysis.explained_variance_ratio_:
    sumi = sumi+i
    n = n+1
    if sumi >= 0.9:
        break
print n


# In[80]:

print analysis.components_[0,0]


# In[81]:

x_origin = analysis.transform(price_data.iloc[:,1:])


# In[71]:

#print x_origin[:,0]


# In[82]:

dj_index = pd.read_csv('djia_index.csv')
#print dj_index.iloc[:,1]


# In[83]:

print np.corrcoef(x_origin[:,0],dj_index.iloc[:,1])


# In[94]:

print np.where(analysis.components_[0] == max(analysis.components_[0]))


# In[107]:

print price_data.columns[27]


# In[ ]:



