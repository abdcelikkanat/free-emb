import numpy as np
from bayespy.nodes import CategoricalMarkovChain
from bayespy.inference import VB
import bayespy.nodes as bayes


print("++++++++++++++++++++++++++")

N = 10000
y = np.random.choice(3, size=N, p=[0.3, 0.6, 0.1])


a0 = [0.5, 0.1, 0.1]

mu0 = -1
lambda0 = 5



#MU = bayes.Gaussian(mu=mu0, Lambda=0.9)
#X = bayes.Gaussian(mu=0.2, Lambda=0.4, plates=(N, ))
P = bayes.Dirichlet(a0)
X = bayes.Categorical(P, plates=(N, ))

#P.initialize_from_random()

Q = VB(X, P)

X.observe(y)
Q.update(repeat=1000)


print(X.pdf([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
print(P.random())
#print(np.sum(y==2))
