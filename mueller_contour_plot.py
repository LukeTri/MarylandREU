from matplotlib import colors

from euler_maruyama import euler_maruyama_white_noise as mp
import numpy as np
import matplotlib.pyplot as plt


def plotMuellerContours():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)

    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))

    fig, ax = plt.subplots(1)
    ax.contour(x, y, v_func(x, y), 1000)
    plt.show()


def plotMuellerContours2():
    v_func = np.vectorize(mp.MuellerPotentialNonVectorized)  # major key!

    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                       np.linspace(-0.5, 2, 100))
    Z = v_func(X, Y)
    print(X)
    tics = np.linspace(-150, 150, 15)
    plt.contour(X, Y, Z, tics,colors='grey',linewidths=1)


    plt.pcolormesh(X, Y, Z, norm=colors.SymLogNorm(linthresh=100),cmap='viridis', shading='gouraud')
    plt.colorbar()

    plt.show()


plotMuellerContours2()
