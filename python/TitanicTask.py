import numpy as np
import pandas
import scipy.stats as stats
import operator

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
print data['Sex'].value_counts()
print data['Survived'].value_counts()
print float(data['Survived'][data['Survived'] == 1].count())/data['Survived'].count()
print float(data['Pclass'][data['Pclass'] == 1].count())/data['Pclass'].count()
print "Mean age: "+str(data['Age'].mean())
print "Median age:"+str(data['Age'].median())
print "Pearson corr: "+str(stats.pearsonr(data['SibSp'], data['Parch']))
print "Pearson corr: "+str(data['SibSp'].corr(data['Parch']))

femaleNames = data['Name'][data['Sex'] == 'female']
namesRating = dict()
for name in femaleNames:
    splitName = str(name).split()[2]
    ##print splitName
    count = namesRating.get(splitName)
    if count is None:
        count = 1
    else:
        count = count + 1
    namesRating[splitName] = count


namesRating = sorted(namesRating.items(), reverse=True, key=operator.itemgetter(1))
print namesRating



