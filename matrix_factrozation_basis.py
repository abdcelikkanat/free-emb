import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx


dim = 128
g = nx.read_gml("./datasets/citeseer_undirected.gml")
#g = nx.read_gml("./datasets/karate.gml")
output_file = "./citeseer_matrix_basis.embedding"

N = g.number_of_nodes()

row = np.array([node for node in range(N) for nb in nx.neighbors(g, str(node))])
col = np.array([int(nb) for node in range(N) for nb in nx.neighbors(g, str(node))])
data = np.ones(shape=row.shape, dtype=np.int)

A = csr_matrix((data, (row, col)), shape=(N, N))


# Generate a basis vectors
basis = np.random.uniform(size=(N, dim), low=-0.5, high=0.5)

basis = basis / np.sqrt(np.sum(basis**2, 0))

for i in range(1, dim):
    v = basis[:, i]
    for j in range(i):
        u = basis[:, j]
        v = v - np.dot(v, u)*u
    basis[:, i] = v / np.sqrt(np.sum(v**2, 0))


P = (A.T / np.sum(A, 1)).T
M = 18.0*A + A*A

#for i in range(100):
#    M = A*M
#    M = M - np.mean(M)

emb = np.dot(M.toarray(), basis)


with open(output_file, 'w') as f:
    f.write("{} {}\n".format(N, dim))

    for node in range(N):
        f.write("{} {}\n".format(node, " ".join(str(round(value, 5)) for value in emb[node, :])))