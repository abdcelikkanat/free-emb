import numpy as np
import networkx as nx
from scipy.stats import norm, multivariate_normal

mu1 = [0.0, 0.0, 0.0]
sigma1 = [[1.2, 0.5, 0.2], [0.5, 1.2, 0.1], [0.2, 0.1, 1.2]]


mu2 = [3.5, 3.5, 3.5]
sigma2 = [[0.5, 0.2, 0.1], [0.2, 0.5, 0.05], [0.1, 0.05, 0.5]]

x1 = np.random.multivariate_normal(mu1, sigma1, 600)
x2 = np.random.multivariate_normal(mu2, sigma2, 400)

x = np.concatenate((x1, x2))

N = x.shape[0]
num_of_classes = 2
dim = len(mu1)



def est_pi(postr):

    return np.sum(postr, 0) / float(N)


def est_mu(x, postr):

    est_mu = np.zeros(shape=(dim, num_of_classes), dtype=np.float)


    for c in range(num_of_classes):
        est_mu[:, c] = np.sum(np.multiply(x.T, postr[:, c]).T, 0) / np.sum(postr[:, c], 0)


    return est_mu


def est_sigma(x, postr, est_mu):

    sigma = np.zeros(shape=(dim, dim, num_of_classes), dtype=np.float)
    postr_sum = np.sum(postr, 0)

    for c in range(num_of_classes):
        for i in range(N):
            sigma[:, :, c] += postr[i, c] * np.outer(x[i, :] - est_mu[:, c], x[i, :] - est_mu[:, c])

        sigma[:, :, c] = sigma[:, :, c] / postr_sum[c]

    return sigma


def get_postr(x, est_pi, est_mu, est_sigma):

    likelihood = np.zeros(shape=(num_of_classes, N), dtype=np.float)
    postr = np.zeros(shape=(N, num_of_classes), dtype=np.float)

    for c in range(num_of_classes):
        likelihood[c, :] = multivariate_normal.pdf(x, mean=est_mu[:, c], cov=est_sigma[:, :, c])
        postr[:, c] = likelihood[c, :] * est_pi[c]

    for i in range(N):
        postr[i, :] = postr[i, :] / np.sum(postr[i, ])

    return postr



postr = np.random.random(size=(N, num_of_classes))
postr = np.divide(postr.T, np.sum(postr, 1)).T


estimated_pi = np.zeros(shape=(num_of_classes), dtype=np.float)
estimated_mu = np.zeros(shape=(dim, num_of_classes), dtype=np.float)
estimated_sigma = np.zeros(shape=(dim, dim, num_of_classes), dtype=np.float)

for _ in range(1000):

    estimated_pi = est_pi(postr)
    estimated_mu = est_mu(x, postr)

    estimated_sigma = est_sigma(x, postr, estimated_mu)
    const_sigma = np.zeros(shape=(dim, dim, num_of_classes), dtype=np.float)
    const_sigma[:, :, 0] = sigma1
    const_sigma[:, :, 1] = sigma2
    #estimated_sigma = const_sigma

    postr = get_postr(x, estimated_pi, estimated_mu, estimated_sigma)

print("------------o-------------")
print(estimated_pi)
print(estimated_sigma[:,:,0])
print(estimated_sigma[:,:,1])

""" """