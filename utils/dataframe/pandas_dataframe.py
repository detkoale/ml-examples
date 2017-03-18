
# coding: utf-8

# # Библиотека Pandas

# ## Data Frame

# In[ ]:

import pandas as pd


# In[ ]:

#создание DataFrame по столбцам с помощью словарей
frame = pd.DataFrame({'numbers':range(10), 'chars':['a']*10})


# In[ ]:

frame


# In[ ]:

#создание DataFrame с помощью чтения данных из файла
frame = pd.read_csv('dataset.tsv', header=0, sep='\t')


# In[ ]:

frame


# In[ ]:

frame.columns


# In[ ]:

frame.shape


# In[ ]:

new_line = {'Name':'Perov', 'Birth':'22.03.1990', 'City':'Penza'}


# In[ ]:

#добавление строки в DataFrame
frame.append(new_line, ignore_index=True)


# In[ ]:

#добавление строки в DataFrame
frame = frame.append(new_line, ignore_index=True)


# In[ ]:

frame


# In[ ]:

#добавление столбца в DataFrame
frame['IsStudent'] = [False]*5 + [True]*2


# In[ ]:

frame


# In[ ]:

#удаление строк DataFrame
frame.drop([5,6], axis=0)


# In[ ]:

frame


# In[ ]:

#удаление строк DataFrame (inplace)
frame.drop([5,6], axis=0, inplace=True)


# In[ ]:

frame


# In[ ]:

#удаление столбца DataFrame (inplace)
frame.drop('IsStudent', axis=1, inplace=True)


# In[ ]:

frame


# In[ ]:

#запись DataFrame в файл
frame.to_csv('updated_dataset.csv', sep=',', header=True, index=False)


# In[ ]:

get_ipython().system(u'cat updated_dataset.csv')


# In[ ]:

#аналог команды для пользователей Windows
get_ipython().system(u'more updated_dataset.csv')

