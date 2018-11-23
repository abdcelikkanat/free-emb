import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

N = 10000
K = 50
epsilon = 25


f = np.random.normal(size=(N, K))



prod = np.dot(f, f.T)

o = np.ones(shape=prod.shape)

z = np.asarray(prod > epsilon, dtype=np.int) +  np.asarray(prod < -epsilon, dtype=np.int)
for i in range(z.shape[0]):
    z[i, i] = 0

deg_seq = np.sum(z, 1)

#print(prod)
#print(z)
#print(deg_seq)
456
plt.figure()
plt.hist(deg_seq, range=(min(deg_seq), max(deg_seq)), bins=[b for b in np.arange(min(deg_seq)-0.5, max(deg_seq)+1.5, 1)])
plt.show()

g = nx.Graph()
for i in range(z.shape[0]):
    for j in range(i+1, z.shape[0]):
        g.add_edge(i, j)

cc = nx.average_clustering(g)

print("Average cc: {}".format(cc3))


plt.figure()
nx.draw(g)
plt.show()
