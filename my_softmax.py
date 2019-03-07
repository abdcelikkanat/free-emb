import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
import stoch_block_model
import community


g = nx.Graph()
g.add_edges_from([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                  [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7],
                  [3, 4] ])

#g = nx.read_gml("./datasets/karate.gml")
#g = nx.read_gml("./datasets/citeseer_undirected.gml")

N = g.number_of_nodes()
dim = 2

nb_list = None

center = None
context = None

def initialize():
    global g, N, dim, center, context, nb_list

    center = np.random.rand(N, dim)
    context = np.random.rand(N, dim)

    nb_list = [[] for _ in range(N)]
    for node in g.nodes():
        for nb in g.neighbors(node):
            nb_list[int(node)].append(int(nb))

def gradient_context(node):
    global nb_list

    gradient = 0.0
    for nb in nb_list[int(node)]:
        gradient += center[nb]







C = 10.0

r = np.random.rand(N, dim)
c = np.random.rand(dim, N)

c_sum = np.sum(c, axis=1)
for inx, r_val in enumerate(r):
    r[inx] *= C / (r_val * c_sum)

print(c_sum)
print(r)