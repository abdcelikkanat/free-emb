import numpy as np
import networkx as nx
import stoch_block_model



sizes = [12, 18]
matrix_p = [[0.85, 0.05],
            [0.05, 0.75]]
g = stoch_block_model.generate(sizes, matrix_p)


#g = nx.Graph()
#g.add_edges_from([[0,1], [0,2], [1,2], [3,4], [4,5], [3,5], [0,3]])

N = g.number_of_nodes()
K = 2





def compute_prob(node, mu, g):

    p = np.zeros(shape=(K, ), dtype=np.float)
    for nb in g.neighbors(node):
        #p += np.multiply(mu[int(node), :], mu[int(nb), :])
        p += mu[int(nb), :]
    #p = np.exp(5.0*mu[int(node), :] + p)
    p = np.exp(0.0*mu[int(node), :] + p)
    p = p / np.sum(p)

    return p


num_iters = 500
mu = np.random.dirichlet(alpha=[0.3, 0.6], size=(N, ))

print("Init= ", np.argmax(mu, axis=1))

for iter in range(num_iters):

    q = np.zeros(shape=(N, K), dtype=np.float)
    for node in g.nodes():
        mu[int(node), :] = compute_prob(node, mu, g)

    #mu = q

print("Last: ", np.argmax(mu, axis=1))




