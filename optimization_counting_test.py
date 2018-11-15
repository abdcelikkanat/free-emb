import numpy as np

K = 100
x = np.random.randint(low=0, high=K, size=1000);

real_m = np.mean(x);

u = np.unique(x)
c = [np.sum(x==k) for k in range(K)]

print(c)

num_of_iters = 100
alpha = 0.001
estimated_mean = 2.0
for iter in range(num_of_iters):

    for k in range(K):
        delta = c[k]*(k - estimated_mean)
        estimated_mean = estimated_mean - alpha*delta
