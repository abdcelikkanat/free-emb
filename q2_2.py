#########################
##  Question 2, part3, ##
#########################

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

data = np.zeros(shape=(0, 3),  dtype=np.float)


for suffix in ['A', 'B', 'C']:
    data = np.concatenate((data, np.loadtxt("./classification_data_HWK1/classification"+suffix+".train", dtype=np.float)))

x = data[:, 0:2]
y = data[:, 2]

N = x.shape[0]  # number of points

# Shuffle data
x, y = shuffle(x, y)

# Insert a dimension to X with 1's
X = np.hstack((np.ones(shape=(N, 1), dtype=np.float), x))

w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


xx1_min, xx2_min = np.amin(X, 0)[1], np.amin(X, 0)[2]
xx1_max, xx2_max = np.amax(X, 0)[1], np.amax(X, 0)[2]


xx2 = np.arange(xx2_min, xx2_max, 0.1)
xx1 = ((0.5 - w[0]) - w[2]*xx2) / w[1]




plt.figure(1)

x0 = x[np.where(y == 0), :][0]
x1 = x[np.where(y == 1), :][0]


plt.plot(x0[:, 0], x0[:, 1], 'b.', x1[:, 0], x1[:, 1], 'rx')

plt.plot(xx1, xx2, 'c.')

plt.show()
