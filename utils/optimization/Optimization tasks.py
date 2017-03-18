
# coding: utf-8

# In[1]:

import numpy as np
import scipy.optimize
from matplotlib import pylab as plt


# In[8]:

def f(x):
    return np.sin(x/5.)*np.exp(x/10.)+5*np.exp(-x/2.)

fmin = scipy.optimize.minimize(f, x0=30., method='BFGS')
print fmin

#print "Minimum:", round(fmin.fun,2)

global_fmin = scipy.optimize.differential_evolution(f, [(1,30)])
print global_fmin


# In[24]:

def h(x):
    return np.int(f(x))


# In[25]:

hmin1 = scipy.optimize.minimize(h, x0=30., method='BFGS')
print hmin1.fun

hmin2 = scipy.optimize.differential_evolution(h, [(1,30)])
print hmin2.fun


# In[14]:

get_ipython().magic(u'matplotlib inline')


# In[ ]:

x = np.arange(1,20,1)
plt.plot(x,h(x))
plt.show()


# In[ ]:



