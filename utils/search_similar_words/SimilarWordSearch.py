import numpy as np
import re

import scipy.spatial

file_name = 'sentences.txt'
file_handler = open(file_name)

words_index = {}
sentences = list()
rows_num = 0
index = 0

# parse file with sentences
for line in file_handler:
    words = re.split('[^a-z]', line.lower())
    words_freq = {}
    freq = 0
# make a dictionary with words from sentence
    for word in words:
        if len(word) == 0: continue

        if words_index.get(word) is None:
            words_index.update({word: index})
            index = index + 1

        freq = words_freq.get(word)

        if freq is None:
            freq = 1
        else:
            freq = freq + 1

        words_freq.update({word:freq})

    sentences.append(words_freq)
    rows_num = rows_num + 1

result_matrix = np.zeros((rows_num, len(words_index)))
row = 0
for sentence in sentences:
    for word in sentence.iterkeys():
        col = words_index.get(word)
        result_matrix[row][col] = int(sentence.get(word))
        print word, 'column:',col, 'matrix value:',result_matrix[row][col]
    row = row+1
print result_matrix

for i in range(1,rows_num):
    print 'row:',i, 'value:',scipy.spatial.distance.cosine(result_matrix[0],result_matrix[i])

