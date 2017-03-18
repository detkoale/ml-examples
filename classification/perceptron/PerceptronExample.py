import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train_data = pandas.read_csv('perceptron-train.csv', header=None)
test_data = pandas.read_csv('perceptron-test.csv', header=None)

y_test = test_data[[0]].values.ravel()
x_test = test_data.iloc[:, 1:]
#x_test = test_data[test_data.columns[1:]]
#print "Y test:", y_test, "X test:",x_test

y_train = train_data[[0]].values.ravel()
x_train = train_data.iloc[:, 1:]

clf = Perceptron(random_state=241)
clf.fit(X=x_train, y=y_train)
y_predict = clf.predict(x_test)

not_scaled_accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
print not_scaled_accuracy

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

clf.fit(X=X_train_scaled, y=y_train)
y_scaled_predict = clf.predict(X_test_scaled)
scaled_accuracy = accuracy_score(y_true=y_test, y_pred=y_scaled_predict)
print scaled_accuracy

print round(scaled_accuracy-not_scaled_accuracy, 3)