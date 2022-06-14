import mueller_potential as mp
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt


def plotMuellerContours():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)  # major key!

    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))

    fig, ax = plt.subplots(1)
    ax.contour(x, y, v_func(x, y), 1000)
    plt.show()


def plotMuellerContours2():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))
    plt.figure()
    Z = v_func(X, Y)
    tics = np.linspace(-150, 150, 30)
    CS = plt.contour(X, Y, Z, tics)
    plt.clabel(CS, inline=False, fontsize=10)
    plt.show()


plotMuellerContours2()
