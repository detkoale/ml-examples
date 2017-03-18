
# coding: utf-8

# #Первое знакомство с NumPy, SciPy и Matplotlib

# ##Numpy

# In[2]:

import numpy as np


# In[3]:

x = [2, 3, 4, 6]
y = np.array(x)


# In[4]:

print type(x), x
print type(y), y


# In[5]:

print x[1:3]


# In[6]:

print y[1:3]


# In[7]:

print x[[0, 2]]


# In[8]:

print y[[0, 2]]


# In[9]:

print y[y>3]


# In[10]:

print x * 5


# In[11]:

print y * 5


# In[12]:

print x ** 2


# In[13]:

print y ** 2


# In[14]:

matrix = [[1, 2, 4], [3, 1, 0]]
nd_array = np.array(matrix)


# In[15]:

print matrix[1][2]


# In[17]:

print nd_array[1, 2]


# In[21]:

print np.random.rand()


# In[23]:

print np.random.randn()


# In[24]:

print np.random.randn(4)


# In[25]:

print np.random.randn(4, 5)


# In[26]:

print np.arange(0, 8, 0.1)


# In[27]:

print range(0, 8, 0.1)


# In[28]:

get_ipython().magic(u'timeit np.arange(0, 10000)')
get_ipython().magic(u'timeit range(0, 10000)')


# ##SciPy

# In[29]:

from scipy import optimize


# In[30]:

def f(x):
    return (x[0] - 3.2) ** 2 + (x[1] - 0.1) ** 2 + 3

print f([3.2, 0.1])


# In[31]:

x_min = optimize.minimize(f, [5, 5])
print x_min


# In[32]:

print x_min.x


# In[33]:

from scipy import linalg


# In[34]:

a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])

x = linalg.solve(a, b)
print x


# In[35]:

print np.dot(a, x)


# In[36]:

X = np.random.randn(4, 3)
U, D, V = linalg.svd(X)
print U.shape, D.shape, V.shape
print type(U), type(D), type(V)


# ##Matplotlib

# In[37]:

get_ipython().magic(u'matplotlib inline')


# In[38]:

from matplotlib import pylab as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()


# In[39]:

x = np.arange(-10, 10, 0.1)
y = x ** 3
plt.plot(x, y)
plt.show()


# ## Все вместе

# In[40]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# In[55]:

x = np.arange(0, 10, 2)
y = np.exp(-x/3.0) + np.random.randn(len(x)) * 0.05

print x[:5]
print y[:5]


# In[56]:

f = interpolate.interp1d(x, y, kind='quadratic')
xnew = np.arange(0, 8, 0.1)
ynew = f(xnew)


# In[57]:

plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()


# In[ ]:



