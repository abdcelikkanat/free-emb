import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_gml("../datasets/citeseer_undirected.gml")

g = max(nx.connected_component_subgraphs(g), key=len)
mapping = {node: str(id) for id, node in enumerate(g.nodes())}
g = nx.relabel_nodes(g, mapping=mapping)


num_of_nodes = g.number_of_nodes()

# perform random walks
L = 10000
N = 1

walks = []
for node in g.nodes():
    walk = [node]

    for l in range(1, L):
        prev = walk[-1]
        nb_list = list(nx.neighbors(g, prev))
        p = np.asarray([float(g.degree(nb)) for nb in nb_list])
        p = p / np.sum(p)
        next = np.random.choice(a=nb_list, size=1, p=p)[0]
        walk.append(next)

    walks.append(walk[1:])


# Degree frequency
expected_freq = np.zeros(shape=(num_of_nodes, ), dtype=np.float)
estimated_freq = np.zeros(shape=(num_of_nodes, ), dtype=np.float)

expected_freq = np.asarray([float(nx.degree(g, str(node))) for node in range(num_of_nodes)])
expected_freq = expected_freq / np.sum(expected_freq)

for walk in walks:
    for w in walk:
        estimated_freq[int(w)] += 1.0
estimated_freq = estimated_freq / np.sum(estimated_freq)


plt.figure()
plt.plot(np.arange(1, num_of_nodes+1), expected_freq, 'r.', label="expected")
plt.plot(np.arange(1, num_of_nodes+1), estimated_freq, 'b.', label="estimated")
plt.show()


plt.figure()
plt.plot(estimated_freq, expected_freq, 'x')
plt.show()