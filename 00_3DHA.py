from datetime import datetime
start_time = datetime.now()

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from scipy import sparse
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import constants as const
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as mticker

# Load interactive widgets
# import ipywidgets as widgets
#
# import ipyvolume as ipv

my_path = os.path.abspath(r"C:\Users\KuChris\Desktop\HENEW")

hbar = const.hbar
e = const.e
m_e = const.m_e
pi = const.pi
epsilon_0 = const.epsilon_0
joul_to_eV = e

n=1-1
l=0
m=0
N = 2000
k=N-1


# N = 2000    # Number of intervals (J=1 in my notes)
# dim = N # Number of internal points
# xl = 0      # xl corresponds to origin
# xr = 200.   #
# delta = (xr-xl)/N
# l=0
# r = np.linspace(xl+delta,xr-delta,dim)


## Radial Solution
##Create r

r = np.linspace(10e-9, 0, N, endpoint=False)
#r = np.linspace(1e-7,,N)

##potential m\deltax^2 unit
p = 'HA'

V = -(e**2) / ((4.0 * pi * epsilon_0)* (r))

##create matrix
#3
diag = np.ones([N])

h = r[1]-r[0]
diags = np.array([diag/h**2, -2*diag/h**2, diag/h**2])
D = sparse.spdiags(diags, np.array([-1, 0,1]), N, N)
I = np.identity(N)

#5
# diag = np.ones([N])
#
# h = r[1]-r[0]
# diags = np.array([-diag/(12*h**2), 16*diag/(12*h**2), -30*diag/(12*h**2), 16*diag/(12*h**2), -diag/(12*h**2)])
# D = sparse.spdiags(diags, np.array([-2,-1, 0,1,2]), N, N)
# I = np.identity(N)

#9
# diag = np.ones([N])
# h = r[1]-r[0]
# diags = np.array([-diag/(560*h**2), 8*diag/(315*h**2), -diag/(5*h**2), 8*diag/(5*h**2), -205*diag/(72*h**2), 8*diag/(5*h**2), -diag/(5*h**2), 8*diag/(315*h**2), -diag/(560*h**2)])
# D = sparse.spdiags(diags, np.array([-4,-3,-2,-1,0,1,2,3,4]), N, N)
# I = np.identity(N)


##define energy
#l = 0
angular = (l * (l + 1))/ r**2

T = D
U = sparse.diags(V.reshape(N),(0))
L = sparse.diags(angular.reshape(N),(0))
H = -((hbar**2)/(2*m_e))*T + U -((hbar**2)/(2*m_e))*L



##Solve for eigenvector and eigenvalue
#k = 20
eigenvalues , eigenvectors = eigsh(H, k, which='SM')
def get_e(n):
    return 1e+10*eigenvectors.T[n].reshape((N))

##number of state
for n in range (0,10):
##plot eigenvector
    fig1 = plt.figure(1)
    plt.plot(r *1e+10, get_e(n))
    plt.xlabel('r ($\\mathrm{\AA}$)')
    plt.ylabel('Eigenfunction ($\\mathrm{\AA}^{-3/2}$)')
    plt.title("Plot of Eigenfunction")
    #plt.title("Plot of Eigenfunction for {} state".format(n))

    at = AnchoredText(r"$l={}$".format(l), prop=dict(size=15), frameon=True, loc=4)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax =plt.gca().add_artist(at)

    plt.tight_layout()
    #plt.savefig(os.path.join(my_path, 'Figure_{}_{}.1.png'.format(l,n)))
    #plt.close()

    ##plot Probability Density
    fig2 = plt.figure(2)
    energies = "E = {: >5.4f} eV".format(eigenvalues[n] / e)
    plt.plot(r *1e+10, get_e(n)**2, label=energies)
    plt.xlabel('r ($\\mathrm{\AA}$)')
    plt.ylabel('probability density ($\\mathrm{\AA}^{-3}$)')
    plt.title("Plot of Probability Density")
    #plt.title("Plot of Probability Density for {} state".format(n))

    at = AnchoredText(r"$l={}$".format(l), prop=dict(size=15), frameon=True, loc=4)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax =plt.gca().add_artist(at)

    plt.tight_layout()
    #plt.savefig(os.path.join(my_path, 'Figure_{}_{}.2.png'.format(l,n)))
    #plt.close()


##plot eigenvalues
fig = plt.figure(3)
alpha = eigenvalues[0]/2
E_a = eigenvalues
b = np.arange(0, len(eigenvalues),1)
plt.scatter(b, E_a/e, s=250, marker="_", linewidth=1, zorder=0)
plt.title("Plot of eigenvalues")
plt.xlabel(r'n')
plt.ylabel(r'$E (eV)$')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

c ="\n".join(map(str,(np.round(eigenvalues/e,3))))

at = AnchoredText(c,prop=dict(size=10), frameon=True, loc='upper right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax =plt.gca().add_artist(at)
plt.tight_layout()
#plt.savefig(os.path.join(my_path, 'Figure_e._{}.png'.format(l)))

#plt.close()

plt.show()

end_time = datetime.now()
## time log
time = end_time - start_time
times = time.total_seconds()
print('Time: {}'.format(times))