import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt





delta = 0.01
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

delta = 0.01
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)

pts = np.vstack((X.flatten(), Y.flatten())).T

M = np.asarray([[15.0, 5], [0.0, 0.00]], dtype=np.float)

z = np.zeros(shape=pts.shape[0], dtype=np.float)
for i in range(pts.shape[0]):
    z[i] = np.exp( -np.dot(np.dot(pts[i, :], M), pts[i, :]) )


print(X.shape)
print(Y.shape)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, np.reshape(z, newshape=X.shape))
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Simplest default with labels')

plt.show()