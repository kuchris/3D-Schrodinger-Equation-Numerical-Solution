from datetime import datetime
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from scipy import sparse
from numba import jit
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

my_path = os.path.abspath(r"C:\Users\KuChris\Desktop\result1")


##Create meshgrid for x y z
for N in range(2,102):
    start_time = datetime.now()
    L = 8
    X,Y,Z= np.meshgrid(np.linspace(-L/2,L/2,N, dtype=float),
        np.linspace(-L/2,L/2,N, dtype=float),
        np.linspace(-L/2,L/2,N, dtype=float))

    ##potential m\deltax^2 unit
    p = 'ISW'

    # ISW
    V = np.zeros([N, N, N])

    # SHO
    #V =0.0001*((X)**2 + (Y)**2 +(Z)**2)/2.0

    #SP
    #from scipy.signal import square
    #V = 200.0*(1.0 - square(2.1*np.pi*np.sqrt((X/L)**2 + (Y/L)**2) + (Z/L)**2))

    ##create matrix
    diag = np.ones([N])
    diags = np.array([diag, -2*diag, diag])
    D = sparse.spdiags(diags, np.array([-1, 0,1]), N, N)
    I = np.identity(N)
    ##define energy
    #D1 = sparse.kronsum(D,D)
    #D2 = sparse.kronsum(D1,D)
    T = -1/2 * (sparse.kron(sparse.kron(D,I),I)
        + sparse.kron(sparse.kron(I,D),I)
        + sparse.kron(sparse.kron(I,I),D))

    #T = -1/2 * D2
    U = sparse.diags(V.reshape(N**3),(0))
    H = T+U

    ##plot V
    fig = plt.figure(0,figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    plot0 = ax.scatter3D(X, Y, Z, c=V,
        cmap=cm.seismic,
        s=0.01,
        alpha=0.5,
        antialiased=True)

    fig.colorbar(plot0, shrink=0.5, aspect=5)
    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Z')
    ax.set_title("Plot of V")
    #plt.rcParams.update({"savefig.facecolor": (0, 0, 0, 0)})
    #plt.savefig(os.path.join(my_path, 'Figure_{}.0.png'.format(p)))
    ##number of state
    for n in range (0,1):
    ##Solve for eigenvector and eigenvalue
        eigenvalues , eigenvectors = eigsh(H, k=n+1, which='SM')
        def get_e(n):
            return eigenvectors.T[n].reshape((N,N,N))
        ##plot eigenvector
        fig = plt.figure(1,figsize=(9,9))
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_axis_off()
        plot1 = ax.scatter3D(X, Y, Z, c=get_e(n),
            cmap=cm.seismic,
            s=0.01,
            alpha=0.5,
            antialiased=True)

        fig.colorbar(plot1, shrink=0.5, aspect=5)
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')
        ax.set_zlabel(r'Z')
        ax.set_title("Plot of Eigenfunction for {} state".format(n))
        #plt.savefig(os.path.join(my_path, 'Figure_{}_{}.1.png'.format(p,n)))
        ##plot probability density
        fig = plt.figure(2,figsize=(9,9))
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_axis_off()
        plot2 = ax.scatter3D(X, Y, Z, c=get_e(n)**2,
            cmap=cm.hot_r,
            s=0.01,
            alpha=0.5,
            antialiased=True)

        fig.colorbar(plot2, shrink=0.5, aspect=5)
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')
        ax.set_zlabel(r'Z')
        ax.set_title("Plot of Probability Density for {} state".format(n))
        #plt.savefig(os.path.join(my_path, 'Figure_{}_{}.2.png'.format(p,n)))
        ##plot eigenvalues
        #plot3 = plt.figure(3)
        #alpha = eigenvalues[0]/2
        #E_a = eigenvalues/alpha
        #b = np.arange(0, len(eigenvalues),1)
        #plt.scatter(b, E_a, s=1444, marker="_", linewidth=2, zorder=3)
        #plt.title("Plot of eigenvalues")
        # plt.xlabel('$(n_{x})^2+(n_{y})^2+(n_{z})^2$')
        # plt.ylabel(r'$mE/\hbar^2$')
        #
        #c = ['$E_{}$'.format(i) for i in range(0,len(eigenvalues))]
        #
        #for i, txt in enumerate(c):
        #    plt.annotate(txt, (np.arange(0, len(eigenvalues),1)[i], E_a[i]), ha="center")
        #
        # #plt.savefig(os.path.join(my_path, 'Figure_{}.3.pdf'.format(n)))
        #
        #plt.show()
        plt.close('all')

    end_time = datetime.now()
    ## time log
    time = end_time - start_time
    times = time.total_seconds()
    print('N:{}, Time: {}'.format(N,times))

    test = '{};{}\n'.format(N,times)

    name = r"Timedata"
    with open(os.path.join(my_path, name+".txt" ), mode='a') as file:
        file.write(test)

## time plot
filename = 'Timedata.txt'
df = pd.read_csv(os.path.join(my_path, filename),sep = '\;', engine='python')
df = df.astype(float)
df.columns.values[0] = 'n'
df.columns.values[1] = 't'
fig = plt.figure(figsize=(9,9))
plot= plt.plot(df.n, df.t)
plt.title("Computation time for different N")
plt.xlabel(r'Second $s$')
plt.ylabel(r'Size of matrix $N$')
plt.show()
