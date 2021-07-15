from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# This is the true unknown function we are trying to approximate
#f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: np.sin(0.25*x**2).flatten()
#f = lambda x: (0.25*(x**2)).flatten()
#f = lambda x: np.sin(x) + np.random.randn(*x.shape) * np.sqrt(0.01).flatten()
#f = lambda x: (np.sin(x) + 0.2*np.cos(3*x)).flatten()
f = lambda x: np.sin(x).flatten()
x = np.arange(-5, 5, 0.1)

plt.plot(x, f(x))
plt.axis([-5, 5, -3, 3])
plt.show()


def kernel(a, b):
    global kernelParameter_l 
    kernelParameter_l= 1.2
    kernelParameter_sigma = 1.0
    sqdist = np.sum(a**2,axis=1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # np.sum( ,axis=1) means adding all elements columnly; .reshap(-1, 1) add one dimension to make (n,) become (n,1)
    return kernelParameter_sigma*np.exp(-.5 * (1/kernelParameter_l) * sqdist)

# Sample some input points and noisy versions of the function evaluated at
# these points. 
N = 5         # number of existing observation points (training points).
n = 100        # number of test points.
s = 0.00005    # noise variance.

X = np.random.uniform(-3, 3, size=(N,1))     # N training points 
y = f(X) + s*np.random.normal(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))     # line 1 

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))   # k_star = kernel(X, Xtest), calculating v := l\k_star
mu = np.dot(Lk.T, np.linalg.solve(L, y))    # \alpha = np.linalg.solve(L, y) 

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)                  # k(x_star, x_star) 
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)   
s = np.sqrt(s2)

# PLOTS:
plt.figure(1, figsize=(12,10))
plt.clf()
plt.plot(X, y, 'o', ms=7, label='data points', color='red', markeredgecolor='black')
plt.plot(Xtest, f(Xtest), color='black', label='true')
plt.gca().fill_between(Xtest.flat, mu-2*s, mu+2*s, color='lightgrey', label='Uncertainity')
plt.plot(Xtest, mu, 'b--', lw=2, label='prediction')
#plt.title('lengthscale = {}'.format(kernelParameter_l))
#plt.annotate('x', xy=(X[2], f(X[2])), xytext=(0,-2),size=14,            arrowprops=dict(arrowstyle="->",linewidth=1.5,color='red',                            connectionstyle="angle3,angleA=0,angleB=-90"))
#plt.annotate('x*', xy=(Xtest[12], mu[12]), xytext=(-2,-2), size=14,            arrowprops=dict(arrowstyle="-|>",linewidth=1.5, color='b',                            connectionstyle="angle3,angleA=0,angleB=-90"))
plt.title('Gaussian process')
plt.xlabel('input x')
plt.ylabel('output y')
plt.legend(loc='best')
plt.tight_layout()
plt.margins(x=0)
plt.show()
#plt.axis([-5, 5, -3, 3])