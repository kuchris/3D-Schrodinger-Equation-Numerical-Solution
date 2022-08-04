import numpy as np
from scipy import constants as const
from scipy import sparse as sparse
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
from scipy.special import lpmv

x = np.linspace(0,1, 100)
#m v x

for v in range (0,5):
    m=0
    p = lpmv(m,v,x)
    plot1 = plt.figure(1)
    plt.plot(x,p)

plt.show()