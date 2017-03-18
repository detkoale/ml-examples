
# coding: utf-8

# In[1]:

import numpy as np
f = np.arange(1, 2, 0.3, dtype=float)


# In[9]:

type(f)


# In[3]:

c = np.array([False, False, True])


# In[8]:

print c
type(c)


# In[7]:

b = np.array([1, 2, 3, 4, 5], dtype=float)
print b
type(b)


# In[6]:

type(b)


# In[10]:

b = np.array(1, 2, 3, 4, 5, dtype=float)
print b
type(b)


# In[11]:

a = np.array([6, 3, -5])
b = np.array([-1, 0, 7])


# In[12]:

import scipy
scipy.spatial.distance.cdist(a[np.newaxis, :], b[np.newaxis, :], metric='euclidean')


# In[13]:

scipy.spatial.distance.cdist(a[:, np.newaxis], b[:, np.newaxis], metric='euclidean')


# In[14]:

np.linalg.norm(a, ord=2) - np.linalg.norm(b, ord=2)


# In[15]:

scipy.spatial.distance.cdist(b, a, metric='euclidean')


# In[16]:

import scipy


# In[17]:

scipy.spatial.distance.cdist(b, a, metric='euclidean')


# In[21]:

a = np.array([8, 10, -1, 0, 0])
a = a[:, np.newaxis]
print a


# In[19]:

a


# In[ ]:



