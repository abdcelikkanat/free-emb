import numpy as np
from bayespy.nodes import CategoricalMarkovChain
from bayespy.inference import VB
import bayespy.nodes as bayes



#A = np.asarray([[0.6, 0.2, 0.2], [0.45, 0.1, 0.45], [0.4, 0.1, 0.5]])
#O = np.asarray([[0.8, 0.2], [0.1, 0.9], [0.9, 0.1]])

A = [[0.8, 0.2], [0.3, 0.7]]
O = [[0.9, 0.1], [0.2, 0.8]]

A = np.asarray(A)
O = np.asarray(O)


N = 5
L = 10
K = O.shape[0]  # number of hidden variables
E = O.shape[1]  # number of observations



y = [[] for _ in range(N)]


for n in range(N):
    s = np.random.choice(a=K, size=1)[0]
    for l in range(L):
        o = np.random.choice(a=E, size=1, p=O[s, :])[0]
        s = np.random.choice(a=K, size=1, p=A[s, :])[0]
        y[n].append(o)



#L = len(y)


p0 = 0.3  # a vector of size K
t0 = 0.2 # a vector of size K
e0 = 0.1

p_param = p0*np.ones(K, dtype=np.float)
p = bayes.Dirichlet(p_param, name='p')

t_param = t0*np.ones(K, dtype=np.float)
T = bayes.Dirichlet(t_param, plates=(K, ), name='T')

e_param = e0*np.ones(E, dtype=np.float)
E = bayes.Dirichlet(e_param, plates=(K, ), name='E')

z = bayes.CategoricalMarkovChain(p, T, states=L, plates=(N, ), name='Z')
x = bayes.Mixture(z, bayes.Categorical, E, plates=(N, L), name='X')

p.initialize_from_random()
T.initialize_from_random()
E.initialize_from_random()


Q = VB(x, z, E, T, p)

x.observe(y)
Q.update(repeat=1000)

print("---------------------")
print(np.array(y[1][:25]))
print(np.argmax(x.parents[0].get_moments()[0][1], axis=1)[:25])
print("---------------------")
for u in z.parents[1].get_moments():
    print(u)
    print("++")
print("zzzzzzz")
print(x.parents[1].get_moments()[0])
#print(x.get_parameters())
print(E)
print("---------------------")
print(x.get_parameters())
print("zzzzzzz")
print("---------------------")
'''
Q.ignore_bound_checks = True

delay = 1
forgetting_rate = 0.5

for iter in range(hmm_number_of_iters):
    # Observe a random mini-batch
    subset = np.random.choice(a=N, size=hmm_subset_size)

    # print(subsets)
    # print()
    # print(subsets[subset])
    Q['X'].observe([y[inx] for inx in subset])
    # Learn intermediate variables
    Q.update('Z')
    #  Set step length
    step = (iter + delay) ** (-forgetting_rate)
    # Stochastic gradient for the global variables
    Q.gradient_step('p', 'T', 'E', scale=step)

'''

likelihood = Q['E'].random()

qp = p.random()
qT = T.random()
qE = E.random()


#print(qT)
#print(qE)


d = bayes.Dirichlet([0.3, 0.7])
n = bayes.Categorical(d)
print(n.parents[0])
print(n.parents[0].get_moments())
f = n.parents[0].get_moments()[0]
print(np.exp(f))
print(n)
print(n.pdf([0, 1]))
print(E)

