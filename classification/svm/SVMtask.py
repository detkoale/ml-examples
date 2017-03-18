import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None)
y = data[[0]].values.ravel()
X = data.iloc[:, 1:]

svmClassifier = SVC(C=100000, kernel='linear', random_state=241)
svmClassifier.fit(X=X, y=y)
print svmClassifier.support_