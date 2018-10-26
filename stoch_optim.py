import numpy as np

np.random.seed(100)
x = np.random.uniform(low=-5.0, high=5.0, size=1000)

s = np.random.choice(x, size=300, replace=True)

x = np.hstack((x, s))

u, c = np.unique(x, return_counts=True)


print(len(x))
print(len(u))

mean = np.mean(x)

u=x
np.random.shuffle(u)

#u = x

est = 1.2
lr = 0.1
for iter in range(100):
    for i in range(len(u)):
        deltaf = -2*(u[i] - est)
        #deltaf *= c[i]
        est = est - lr*deltaf

    lr = lr / (1.0 + iter*1.0)
    if lr < 0.0001:
        lr = 0.0001
    print("lr: {}".format(lr))
    print("Correct: {} Estimation: {} Error(%): {}".format(mean, est, (est-mean)/mean*100))


