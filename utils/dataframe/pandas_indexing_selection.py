
# coding: utf-8

# # Библиотека Pandas

# ## Data Frame

# In[2]:

import pandas as pd


# In[3]:

#создание DataFrame с помощью чтения данных из файла
frame = pd.read_csv('data_sample_example.tsv', header=0, sep='\t')


# In[4]:

frame


# In[8]:

frame.dtypes


# In[6]:

#изменение типа столбца с помощью функции apply
frame.Birth = frame.Birth.apply(pd.to_datetime)


# In[7]:

frame


# In[ ]:

frame.dtypes


# In[10]:

frame.info()


# In[11]:

#заполнение пропущенных значений с помощью метода fillna
frame.fillna('разнорабочий')


# In[12]:

#заполнение пропущенных значений с помощью метода fillna (inplace)
frame.fillna('разнорабочий', inplace=True)


# In[ ]:

frame


# In[ ]:

frame.Position


# In[ ]:

frame[['Position']]


# In[ ]:

frame[['Name', 'Position']]


# In[ ]:

frame[:3] #выбираем первые три записи


# In[ ]:

frame[-3:] #выбираем три послдение записи


# In[ ]:

frame.loc[[0,1,2], ["Name", "City"]] #работает на основе имен


# In[ ]:

frame.iloc[[1,3,5], [0,1]] #работает на основе позиций


# In[ ]:

frame.ix[[0,1,2], ["Name", "City"]]  #поддерживает и имена и позиции (пример с именами)


# In[ ]:

frame.ix[[0,1,2], [0,1]] #поддерживает и имена и позиции (пример с позициями)


# In[ ]:

#выбираем строки, которые удовлетворяют условию frame.Birth >= pd.datetime(1985,1,1)
frame[frame.Birth >= pd.datetime(1985,1,1)]


# In[ ]:

#выбираем строки, удовлетворяющие пересечению условий
frame[(frame.Birth >= pd.datetime(1985,1,1)) &
      (frame.City != 'Москва')]


# In[ ]:

#выбираем строки, удовлетворяющие объединению условий
frame[(frame.Birth >= pd.datetime(1985,1,1)) |
      (frame.City == 'Волгоград')]

