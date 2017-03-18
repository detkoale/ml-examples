import numpy as np
import pandas
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data', header=None)
x = data.ix[:,1:]
y = data[0]

kf = KFold(n=len(x), random_state=42, n_folds=5, shuffle=True)

#for train_index, test_index in kf:
#    print ("Train:", train_index, "Test:", test_index)

kMeans = list()
for k in range(1,51,1):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(x,y)
    scores = cross_val_score(estimator=kn, X=x, y=y, cv=kf, scoring="accuracy")
    m = scores.mean()
    kMeans.append(m)

m = max(kMeans)
indices = [i for i, j in enumerate(kMeans, 1) if j == m]

print indices[0]
print np.round(m,decimals=2)

X_scale = scale(x)

kMeans = list()
for k in range(1,51,1):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(x,y)
    scores = cross_val_score(estimator=kn, X=X_scale, y=y, cv=kf, scoring="accuracy")
    m = scores.mean()
    #print("K=",k," Score:",m)
    kMeans.append(m)

m = max(kMeans)
indices = [i for i, j in enumerate(kMeans, start=1) if j == m]

print indices[0]
print np.round(m, decimals=2)