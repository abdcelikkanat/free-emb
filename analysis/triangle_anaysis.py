import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

########################################################
def get_nb_comm(g, node2comm):

    counter = 0.0

    for i, j in g.edges():
        if cmp(node2comm[i], node2comm[j]) == 0:
            counter += 1.0

    return 100.0 * (counter / g.number_of_edges())
########################################################


########################################################
def get_only2hop_comm(g, node2comm):

    true_counter = 0.0
    total_counter = 0.0

    for node in g.nodes():
        hop2list = []
        for nb in nx.neighbors(g, node):
            for nb_nb in nx.neighbors(g, nb):
                if nb_nb not in nx.neighbors(g, node):
                    if nb_nb not in hop2list:
                        if node2comm[node] == node2comm[nb_nb]:
                           true_counter += 1.0

                        hop2list.append(nb_nb)
                        total_counter += 1.0

    return 100.0 * (true_counter / total_counter)

########################################################


########################################################
def get_triangles(g, node2comm):
    triangle_count = 0.0
    true_count = 0.0

    for i in g.nodes():
        for j in nx.neighbors(g, i):
            for k in nx.neighbors(g, j):
                for w in nx.neighbors(g, k):
                    if i == w:
                        triangle_count += 1.0
                        if cmp(node2comm[i], node2comm[j]) == 0 and cmp(node2comm[i], node2comm[k]) == 0:
                            true_count += 1.0

    triangle_count /= 6.0
    true_count /= 6.0

    return (true_count / triangle_count)*100.0
########################################################


########################################################
def get_myego(g, node):

    N = g.number_of_nodes()
    min_nb = min(nx.neighbors(g, node), key=g.degree)

    bint = np.zeros(shape=(N, ), dtype=np.int)
    bout = np.zeros(shape=(N, ), dtype=np.int)

    bint[int(node)] = 1
    bint[int(min_nb)] = 1

    for nb in nx.neighbors(g, node):
        bout[int(nb)] = 1
    for nb in nx.neighbors(g, min_nb):
        bout[int(nb)] = 1

    bout = bout - bint


    print(np.where(bout==0))

########################################################

'''

percent = get_only2hop_comm(g, node2comm)
print(percent)
'''


tri = get_triangles(g, node2comm)
print("Tri percent: {}".format(tri))

'''
nb_an = get_nb_comm(g, node2comm)

print(nb_an)
print(tri)



get_myego(g, node='120')
'''


