import numpy as np
import scipy.linalg
from matplotlib import pylab as plt


def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5 * np.exp(-x / 2.)

a = np.array([[1, 1, 1, 1], [1, 4, 4**2, 4**3],
              [1, 10, 10**2, 10**3], [1, 15, 15**2, 15**3]])
b = np.array([f(1), f(4), f(10), f(15)])
c = scipy.linalg.solve(a, b)

print c

xtest = np.arange(1., 15., 0.1)
ytest = c[0] + c[1] * xtest + c[2] * (xtest**2) + c[3] * (xtest**3)
plt.plot(xtest, f(xtest), xtest, ytest)
plt.show()
