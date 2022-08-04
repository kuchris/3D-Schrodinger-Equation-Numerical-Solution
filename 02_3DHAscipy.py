import matplotlib.pyplot as plt

from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy.integrate as integrate
# Import special functions
import scipy.special as spe
plt.ion()

n=8
l=7
m=0

def psi_R(r,n,l):

    coeff = np.sqrt((2.0/n)**3 * spe.factorial(n-l-1) /(2.0*n*spe.factorial(n+l)))

    laguerre = spe.assoc_laguerre(2.0*r/n,n-l-1,2*l+1)

    return coeff * np.exp(-r/n) * (2.0*r/n)**l * laguerre

r = np.linspace(0,100,10000)

R = psi_R(r,n,l)

plt.plot(r, R**2, lw=3)

plt.xlabel('$r [a_0]$',fontsize=20)

plt.ylabel('$R_{nl}(r)$', fontsize=20)

plt.grid('True')

plt.show()

##

def psi_ang(phi,theta,l,m):

    sphHarm = spe.sph_harm(m,l,phi,theta)

    return sphHarm.real

phi, theta = np.linspace(0, np.pi, 100), np.linspace(0, 2*np.pi, 100)

phi, theta = np.meshgrid(phi, theta)

Ylm = psi_ang(theta,phi,l,m)
rho = np.abs(Ylm)**2

x = np.sin(phi) * np.cos(theta) * abs(Ylm)
y = np.sin(phi) * np.sin(theta) * abs(Ylm)
z = np.cos(phi) * abs(Ylm)

'''Set up the 3D Canvas'''

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

''' Normalize color bar to [0,1] scale'''

#fcolors = (Ylm - Ylm.min())/(Ylm.max() - Ylm.min())

'''Make 3D plot of real part of spherical harmonic'''
color_map = cm.jet
scalarMap = cm.ScalarMappable(norm=plt.Normalize(vmin=np.min(rho),vmax=np.max(rho)),cmap=color_map)
C = scalarMap.to_rgba(rho)
ax.plot_surface(x, y, z, facecolors=C, alpha=0.3)

#ax.plot_surface(x, y, z, facecolors=cm.seismic(fcolors), alpha=0.3)


''' Project 3D plot onto 2D planes'''

# cset = ax.contour(x, y, z,20, zdir='z',offset = -1, cmap='summer')
# cset = ax.contour(x, y, z,20, zdir='y',offset =  1, cmap='winter' )
# cset = ax.contour(x, y, z,20, zdir='x',offset = -1, cmap='autumn')


''' Set axes limit to keep aspect ratio 1:1:1 '''

# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
plt.show()

##
def HFunc(r,theta,phi,n,l,m):
    '''
    Hydrogen wavefunction // a_0 = 1

    INPUT
        r: Radial coordinate
        theta: Polar coordinate
        phi: Azimuthal coordinate
        n: Principle quantum number
        l: Angular momentum quantum number
        m: Magnetic quantum number

    OUTPUT
        Value of wavefunction
    '''


    return psi_R(r,n,l) * psi_ang(phi,theta,l,m)

# plt.figure(4)
#plt.plot(r,psi_R(r,n,l))
#plt.plot(r,HFunc(r,theta,phi,n,l,m))
#plt.plot(r,psi_ang(phi,theta,l,m))

plt.figure(figsize=(10,8))


limit = 4*(n+l)

x_1d = np.linspace(-limit,limit,500)
z_1d = np.linspace(-limit,limit,500)
x,z = np.meshgrid(x_1d,z_1d)
y   = 0

r     = np.sqrt(x**2 + y**2 + z**2)
theta = np.arctan2(np.sqrt(x**2+y**2), z )
phi   = np.arctan2(y, x)


psi_nlm = HFunc(r,theta,phi,n,l,m)**2

plt.pcolormesh(x, z, psi_nlm, cmap='inferno')  # Try cmap = inferno, rainbow, autumn, summer,

plt.contourf(x, z,  psi_nlm, 20, cmap='seismic', alpha=0.6)  # Classic orbitals'seismic'

plt.colorbar()

plt.title(f"$n,l,m={n,l,m}$",fontsize=20)
plt.xlabel('X',fontsize=20)
plt.ylabel('Z',fontsize=20)


#plt.plot(r,HFunc(r,theta,phi,n,l,m))
#plt.plot(r,psi_ang(phi,theta,l,m))

plt.show()