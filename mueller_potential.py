import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt


def getNextIteration(x_0, h, b = 1/20):
    xtemp = np.random.normal() * np.sqrt(2 * b**-1 * h)
    ytemp = np.random.normal() * np.sqrt(2 * b**-1 * h)
    return x_0 - nd.Gradient(MuellerPotential)([x_0])*h + np.array([xtemp, ytemp])


def MuellerPotential(x, a = np.array([-1,-1,-6.5,0.7]), b = np.array([0,0,11,0.6]), c = np.array([-10,-10,-6.5,0.7]),
                     d = np.array([-200,-100,-170,15]), z = np.array([1,0,-0.5,-1]), Y = np.array([0,0.5,1.5,1])):
    ret = 0
    for i in range(0,4):
        ret += d[i] * np.e ** (a[i] * (x[0] - z[i])**2 + b[i] * (x[0] - z[i])*(x[1] - Y[i]) + c[i] * (x[1] - Y[i])**2)
    return ret


x = np.array([0, 0])
X = np.zeros(1000)
Y = np.zeros(1000)

for i in range(1000):
    h = 10 ** -5
    x = getNextIteration(x, h)
    X[i] = x[0]
    Y[i] = x[1]
    print(x)

plt.scatter(X, Y)
plt.show()