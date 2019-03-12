import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import community as louvain

#g = nx.read_gml("../datasets/citeseer_undirected.gml")
g = nx.read_gml("../datasets/blogcatalog.gml")

gcc = max(nx.connected_component_subgraphs(g), key=max)

print("Number of connected components: {}".format(nx.number_connected_components(g)))
print("Ratio of the gcc: {}".format(100.0*float(gcc.number_of_nodes())/g.number_of_nodes()))


N = g.number_of_nodes()

########################################################
node2comm = nx.get_node_attributes(g, 'community')
for node in node2comm:
    if type(node2comm[node]) is int:
        node2comm[node] = [node2comm[node]]
########################################################

########################################################
K = 0
for node in node2comm:
    m = max(node2comm[node])
    if K < m:
        K = m
K += 1
########################################################


# Detect communities
c = louvain.best_partition(g)

louvain_K = 0
for node in g.nodes():
    if c[node] > louvain_K:
        louvain_K = c[node]
louvain_K += 1

louvain_comms = {k: [] for k in range(louvain_K)}
for node in g.nodes():
    louvain_comms[c[node]].append(node)

success = 0.0
for k in range(louvain_K):
    louvain_comm_counts = np.zeros(shape=(K, ), dtype=np.int)
    for node in louvain_comms[k]:
        louvain_comm_counts[node2comm[node]] += 1

    print(louvain_comm_counts)

    success += np.max(louvain_comm_counts)

print(100*success/g.number_of_nodes())