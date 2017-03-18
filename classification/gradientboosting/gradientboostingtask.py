import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('gbm-data.csv')
# 1 column contains activity values aka y
y = data.iloc[:,0].values
# least columns contains various features
X = data.iloc[:,1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8,
                                                    random_state = 241)


def sigmoid(y_pred):
    return 1.0/(1.0+np.exp(-y_pred))


def log_loss_result(model, X, y):
    results = []
    for pred in model.staged_decision_function(X):
        results.append(log_loss(y,[sigmoid(y_pred) for y_pred in pred]))
    return results

def min_value_index(selection):
    min_value = min(selection)
    min_index = selection.index(min_value)
    return min_value, min_index

def plot_loss(train_loss, test_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

def model_test(rate):
    model = GradientBoostingClassifier(learning_rate=rate, random_state=241, verbose=True, n_estimators=250)
    model.fit(X_train, y_train)

    train_loss = log_loss_result(model, X_train, y_train)
    test_loss = log_loss_result(model, X_test, y_test)
    plot_loss(train_loss, test_loss)
    return min_value_index(test_loss)

learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
min_loss_results = {}
for rate in learning_rate:
    min_loss_results[rate] = model_test(rate)

print min_loss_results

min_loss_value,min_loss_index = min_loss_results[0.2]
rfc = RandomForestClassifier(random_state=241, n_estimators=min_loss_index)
rfc.fit(X_train, y_train)
y_pred = rfc.predict_proba(X_test)
print round(log_loss(y_test, y_pred),2)