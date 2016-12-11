import numpy as np
import scipy.optimize
from matplotlib import pylab as plt

def f(x):
    return np.sin(x/5.)*np.exp(x/10.)+5*np.exp(-x/2.)

fmin = scipy.optimize.minimize(f, x0=30., method='BFGS')
print fmin
print "Minimum:", round(fmin.fun,2)

global_fmin = scipy.optimize.differential_evolution(f, [(1,30)])
print global_fmin


x = np.arange(1,30,1)
plt.figure(1)
plt.plot(x, f(x))
plt.plot(fmin.x, fmin.fun,'ro')
plt.show()