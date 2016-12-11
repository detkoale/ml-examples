import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
sample = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna(axis=0)


x = sample[['Pclass', 'Fare', 'Age', 'Sex']]
x.loc[:,'Sex'].replace(['male','female'], [1,0], inplace=True)
y = sample['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(x,y)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, x.columns[idx], importances[idx]))

# select 4 features column in initial data
# print data[['Pclass', 'Fare', 'Age', 'Sex']].dropna()

# print data.dtypes
# print data['Sex'].astype('category', categories = [0,1])
