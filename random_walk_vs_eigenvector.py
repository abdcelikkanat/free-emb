import networkx as nx
import numpy as np


P = np.asarray([[0.4, 0.3, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.5, 0.1], [0.3, 0.1, 0.05, 0.05, 0.5],
                [0.3, 0.3, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.4, 0.3]])

A = P.copy()
for _ in range(300):
    A = np.dot(P, A)

#print(np.linalg.eig(P)[1][:, 0])


eigvec = np.real(np.linalg.eig(P.T)[1][:, 0])
eigvec = eigvec / np.sum(eigvec)
print("eigevec: {}".format(eigvec))
print("dot: {}".format(np.dot(eigvec, P)))

print(A[1, :])

def perfom_a_walk(P, node, n, l):

    K = P.shape[0]
    counts = np.zeros(shape=(K, ), dtype=np.float)

    walks = []
    for _ in range(n):
        walk = [node]

        for _ in range(1, l):
            current = walk[-1]
            walk.append(np.random.choice(a=range(K), size=1, p=P[current, :])[0])

        walks.append(walk)

        for w in walk[1:]:
            counts[w] += 1.0

    return counts

counts = perfom_a_walk(P, node=0, n=10, l=3)
print(counts / np.sum(counts))