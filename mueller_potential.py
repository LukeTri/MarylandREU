import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math

MUELLERMIN1 = np.array([0.62347076, 0.02807048])
MUELLERMIN2 = np.array([-0.04997089, 0.46671412])
MUELLERMIN3 = np.array([-0.55821361, 1.44174872])
mumins = np.zeros((3, 2))
mumins[0] = MUELLERMIN1
mumins[1] = MUELLERMIN2
mumins[2] = MUELLERMIN3


def getNextIteration(x_0, h, b=1 / 20):
    xtemp = np.random.normal() * np.sqrt(2 * b ** -1 * h)
    ytemp = np.random.normal() * np.sqrt(2 * b ** -1 * h)
    return x_0 - MuellerPotentialGradient(x_0) * h + np.array([xtemp, ytemp])


def MuellerPotential(x, a=np.array([-1, -1, -6.5, 0.7]), b=np.array([0, 0, 11, 0.6]), c=np.array([-10, -10, -6.5, 0.7]),
                     d=np.array([-200, -100, -170, 15]), z=np.array([1, 0, -0.5, -1]), Y=np.array([0, 0.5, 1.5, 1])):
    ret = 0
    for i in range(0, 4):
        ret += d[i] * np.e ** (
                    a[i] * (x[0] - z[i]) ** 2 + b[i] * (x[0] - z[i]) * (x[1] - Y[i]) + c[i] * (x[1] - Y[i]) ** 2)
    return ret


def MuellerPotentialNonVectorized(x, y, a=np.array([-1, -1, -6.5, 0.7]), b=np.array([0, 0, 11, 0.6]),
                                  c=np.array([-10, -10, -6.5, 0.7]),
                                  d=np.array([-200, -100, -170, 15]), z=np.array([1, 0, -0.5, -1]),
                                  Y=np.array([0, 0.5, 1.5, 1])):
    ret = 0
    for i in range(0, 4):
        ret += d[i] * np.e ** (a[i] * (x - z[i]) ** 2 + b[i] * (x - z[i]) * (y - Y[i]) + c[i] * (y - Y[i]) ** 2)
    return ret


def MuellerPotentialGradient(x, a=np.array([-1, -1, -6.5, 0.7]), b=np.array([0, 0, 11, 0.6]),
                             c=np.array([-10, -10, -6.5, 0.7]),
                             d=np.array([-200, -100, -170, 15]), X=np.array([1, 0, -0.5, -1]),
                             Y=np.array([0, 0.5, 1.5, 1])):
    U_1 = 0
    U_2 = 0
    for i in range(4):
        V = d[i] * np.e ** (
                    a[i] * (x[0] - X[i]) ** 2 + b[i] * (x[0] - X[i]) * (x[1] - Y[i]) + c[i] * (x[1] - Y[i]) ** 2)
        U_1 += (2 * a[i] * (x[0] - X[i]) + b[i] * (x[1] - Y[i])) * V
        U_2 += (b[i] * (x[0] - X[i]) + 2 * c[i] * (x[1] - Y[i])) * V
    return np.array([U_1, U_2])


def getMuellerMinima(x):
    x = fmin(MuellerPotential, x)
    print(x)


def getFirstMinimum(x, h, delta):
    while True:
        x = getNextIteration(x, h)
        for i in range(3):
            if np.linalg.norm(x - mumins[i]) < delta:
                return i

for i in range(10):
    print(getFirstMinimum(np.array([0.2, 0.3]), 10 ** -5, 0.1))
