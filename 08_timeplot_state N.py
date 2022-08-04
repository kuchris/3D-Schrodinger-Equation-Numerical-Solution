import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText
my_path = os.path.abspath(r"C:\Users\KuChris\Desktop\result1") #

#df = pd.read_csv(r"C:\Users\KuChris\Desktop\jobdata\testdate1.csv",
#    sep = '\,',
#    engine='python')

filename = 'Timedatak.txt'

df = pd.read_csv(os.path.join(my_path, filename),sep = '\;', engine='python')

#df.columns = df.iloc[0]
#df.drop(index=0, inplace=True)

#df1 = df.dropna(axis=0, how='any')

#df1 = df1.apply(lambda x: x.str.replace('"',''))

df = df.astype(float)

df.columns.values[0] = 'n'
df.columns.values[1] = 't'
#df['n'] = df.index
#df2.columns.values[1] = 'y'

#df3 = df2.drop(df2[(df2.x > 2) & (df2.x < 3)].index)

fig = plt.figure(0)
fig= plt.plot(df.n, df.t)
#df.plot('x','n')
plt.title("Computation time for different k")
plt.ylabel(r'Second $s$')
plt.xlabel(r'Number of $k$')


##y=a exp(bx)
def linear(x, a, b):
    return a*x+b


x = df.n
y = df.t
fig1= plt.figure(1)
#fig1 = plt.scatter(x,y)
ax1 = plt.gca()
ax1.scatter(x ,y,c='blue',s=10)
#ax1.set_yscale('log')

pars, cov = curve_fit(f=linear, xdata=x, ydata=y, p0=[0, 0], bounds=(-np.inf, np.inf))

# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))

# Calculate the residuals
res = y - linear(x, *pars)

ax1.plot(x, linear(x, *pars), linestyle='--', linewidth=2, color='black')
plt.title("Computation time for different k")
plt.ylabel(r'Second $s$')
plt.xlabel(r'Number of $k$')
#legend1=plt.legend([''], loc =1)
#ax = plt.gca().add_artist(legend1)
a=pars[0]
b=pars[1]
at = AnchoredText(r"$t={:.2f}*k+{:.2f}$" "\n" "$N=50$".format(a, b), prop=dict(size=15), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax =plt.gca().add_artist(at)

plt.tight_layout()
plt.show()
