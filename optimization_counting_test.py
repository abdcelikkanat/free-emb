import numpy as np

K = 100
x = np.random.randint(low=0, high=K, size=10000);

real_m = np.mean(x)

u = np.unique(x)
c = [np.sum(x==k) for k in range(K)]

print(c)

num_of_iters = 10000
alpha = 0.000001
estimated_mean = 2.0
for iter in range(num_of_iters):

    print("Iter: {} mean: {}".format(iter, estimated_mean))

    for k in range(K):
        delta = c[k]*float(estimated_mean - k)
        estimated_mean = estimated_mean - alpha*delta

print("Correct mean: {}".format(real_m))