import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_gml("../datasets/citeseer_undirected.gml")

N = g.number_of_nodes()

#################################################################################
node2comm = nx.get_node_attributes(g, 'community')
for node in node2comm:
    node2comm[node] = [node2comm[node]]
#################################################################################

#################################################################################
K = 0
for node in node2comm:
    m = max(node2comm[node])
    if K < m:
        K = m
K += 1
#################################################################################


def propogate(g, node, n):

    node_list = [nb for nb in nx.neighbors(g, node)]
    for nb in node_list:
        for nb_nb in nx.neighbors(g, nb):
            if nb_nb not in node_list:
                node_list.append(nb_nb)

    node2inf = {}
    subg = nx.subgraph(g, node_list)

    walks = []
    for nn in range(n):
        h = subg.copy()
        walk = [node]

        current = node
        while nx.degree(h, current) != 0:
            next = np.random.choice(a=list(nx.neighbors(h, current)))
            walk.append(next)

            h.remove_edge(current, next)

            current = next

        walks.append(walk)


    counts = {}
    for walk in walks:
        for w in walk:
            if w not in counts:
                counts[w] = 1
            else:
                counts[w] += 1


    return walks, counts


walks, counts = propogate(g, node=u'10', n=5)

print(walks)
print(counts)