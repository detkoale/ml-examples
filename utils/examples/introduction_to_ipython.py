
# coding: utf-8

# In[ ]:

'hello, world!'


# In[2]:

t = 'hello, world!'


# In[3]:

t


# In[4]:

print t


# In[5]:

6**4


# In[6]:

100/12


# In[7]:

100./12


# In[8]:

round(100./12, 3)


# In[9]:

from math import factorial


# In[10]:

factorial(3)


# In[11]:

factorial(10)*0.5


# text

# # Header

# для редактирования формулы ниже использует синтаксис tex

# $$ c = \sqrt{a^2 + b^2}$$

# In[38]:

get_ipython().system(u" echo 'hello, world!'")


# In[39]:

get_ipython().system(u'echo $t')


# In[37]:

get_ipython().run_cell_magic(u'bash', u'', u'mkdir test_directory\ncd test_directory/\nls  -a')


# In[42]:

#удаление директории, если она не нужна
get_ipython().system(u' rm -r test_directory')


# Ниже аналоги команд для пользователей Windows:

# In[ ]:

get_ipython().run_cell_magic(u'cmd', u'', u'mkdir test_directory\ncd test_directory\ndir')


# удаление директории, если она не нужна (windows)

# In[ ]:

get_ipython().run_cell_magic(u'cmd', u'', u'rmdir test_directiory')


# In[23]:

get_ipython().magic(u'lsmagic')


# In[24]:

get_ipython().magic(u'pylab inline')


# In[25]:

y = range(11)


# In[26]:

y


# In[40]:

plot(y)

