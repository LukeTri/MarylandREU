import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt


def getNextIteration(x_0, h, b = 1/20):
    xtemp = np.random.normal() * np.sqrt(2 * b**-1 * h)
    ytemp = np.random.normal() * np.sqrt(2 * b**-1 * h)
    return x_0 - MuellerPotentialGradient(x_0)*h + np.array([xtemp, ytemp])


def MuellerPotential(x, a = np.array([-1,-1,-6.5,0.7]), b = np.array([0,0,11,0.6]), c = np.array([-10,-10,-6.5,0.7]),
                     d = np.array([-200,-100,-170,15]), z = np.array([1,0,-0.5,-1]), Y = np.array([0,0.5,1.5,1])):
    ret = 0
    for i in range(0,4):
        ret += d[i] * np.e ** (a[i] * (x[0] - z[i])**2 + b[i] * (x[0] - z[i])*(x[1] - Y[i]) + c[i] * (x[1] - Y[i])**2)
    return ret

def MuellerPotentialNonVectorized(x, y, a = np.array([-1,-1,-6.5,0.7]), b = np.array([0,0,11,0.6]), c = np.array([-10,-10,-6.5,0.7]),
                     d = np.array([-200,-100,-170,15]), z = np.array([1,0,-0.5,-1]), Y = np.array([0,0.5,1.5,1])):
    ret = 0
    for i in range(0,4):
        ret += d[i] * np.e ** (a[i] * (x - z[i])**2 + b[i] * (x - z[i])*(y - Y[i]) + c[i] * (y - Y[i])**2)
    return ret

def MuellerPotentialGradient(x, a = np.array([-1,-1,-6.5,0.7]), b = np.array([0,0,11,0.6]), c = np.array([-10,-10,-6.5,0.7]),
                     d = np.array([-200,-100,-170,15]), X = np.array([1,0,-0.5,-1]), Y = np.array([0,0.5,1.5,1])):
    U_1 = 0
    U_2 = 0
    for i in range(4):
        V = d[i] * np.e ** (a[i] * (x[0] - X[i]) ** 2 + b[i] * (x[0] - X[i]) * (x[1] - Y[i]) + c[i] * (x[1] - Y[i]) ** 2)
        U_1 += (2 * a[i] * (x[0] - X[i]) + b[i] * (x[1] - Y[i])) * V
        U_2 += (b[i] * (x[0] - X[i]) + 2 * c[i] * (x[1] - Y[i])) * V
    return np.array([U_1, U_2])

def createGraph(x, h, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in range(n):
        x = getNextIteration(x, h)
        X[i] = x[0]
        Y[i] = x[1]

    plt.scatter(X, Y)
    plt.show()

def plotMuellerContours():
    v_func = np.vectorize(MuellerPotentialNonVectorized)  # major key!

    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))

    fig, ax = plt.subplots(1)
    ax.contour(x, y, v_func(x, y), 1000)
    plt.show()

def plotMuellerContours2():
    v_func = np.vectorize(MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))
    plt.figure()
    Z = v_func(X, Y)
    tics = np.zeros(30)
    for i in range(30):
        tics[i] = -150 + 10*i
    CS = plt.contour(X, Y, Z, tics)
    plt.clabel(CS, inline=False, fontsize=10)
    plt.show()

def getFirstMinimum(x, h):
    while True:
        x = getNextIteration(x, h)


plotMuellerContours2()
#createGraph(np.array([0,0]), 10**-5, 100000)