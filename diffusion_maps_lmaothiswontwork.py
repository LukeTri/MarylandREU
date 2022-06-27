import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sigma = 0.2


def diffusion_map(index, lam, r, eigvals):
    ret = np.zeros(len(eigvals))
    i = 0
    t = np.ceil(np.real(np.log(sigma) / (np.log(lam[eigvals[len(eigvals)-1]]) - np.log(lam[eigvals[0]]))))
    for eigenval_num in eigvals:
        ret[i] = lam[eigenval_num]**t * r[index][eigenval_num]
        i += 1
    return ret

def MakeSpiral():
    t = np.linspace(0,3*np.pi,70)
    t2 = np.linspace(0,15,15)
    x = np.zeros((len(t)*len(t2)))
    y = np.zeros(len(t)*len(t2))
    z = np.zeros(len(t)*len(t2))
    X = np.zeros((len(t)*len(t2), 3))
    c = np.zeros(len(t)*len(t2))

    for i in range(len(t)):
        for j in range(len(t2)):
            x[i * len(t2) + j] = t[i] * np.sin(t[i])
            y[i * len(t2) + j] = t2[j]
            z[i * len(t2) + j] = t[i] * np.cos(t[i])
            c[i * len(t2) + j] = t[i]
            X[i * len(t2) + j] = np.array([x[i * len(t2) + j], y[i * len(t2) + j], z[i * len(t2) + j]])

    dist = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            dist[i][j] = np.linalg.norm(X[i]-X[j])

    drowmin = np.zeros(len(dist))
    for i in range(len(dist)):
        m = 1000
        for j in range(len(dist)):
            if i != j and dist[i][j] < m:
                drowmin[i] = dist[i][j]
    epsilon = 5

    k = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            k[i][j] = np.exp(-(dist[i][j])**2 / epsilon)

    q = np.zeros(len(X))
    P = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        q[i] = np.sum(k[i])
        P[i] = k[i] / q[i]

    w, v = np.linalg.eig(P)

    print(v)


    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    lam = np.diag(w)

    pi = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        pi[i][i] = q[i] / np.sum(q)

    v_tran = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            v_tran[i][j] = v[j][i]

    scaled_identity = v_tran @ pi @ v
    print(scaled_identity)

    for i in range(len(X)):
        for j in range(len(X)):
            v[i][j] = v[i][j] / np.sqrt(scaled_identity[j][j])
            v_tran[j][i] = v_tran[j][i] / np.sqrt(scaled_identity[j][j])

    print(v[:,0])


    print(v_tran @ pi @ v)

    newX = np.zeros((len(X), 3))

    for i in range(len(X)):
        newX[i] = diffusion_map(i, w, v, np.array([1,2,3]))

    x = newX[:, 0]
    y = newX[:, 1]
    z = newX[:, 2]
    ax.scatter(x, y, z, c=c)

    test = np.linspace(0, len(w), len(w))
    #ax.plot(test, w)

    plt.show()



MakeSpiral()
