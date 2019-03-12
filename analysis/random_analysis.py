import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

g = nx.read_gml("../datasets/citeseer_undirected.gml")
#g = nx.read_gml("../datasets/blogcatalog.gml")

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


def find_node_in_comm(g, node2comm, comm_label):

    node_list = list(g.nodes())
    np.random.shuffle(node_list)

    for node in node_list:

        if node2comm[node][0] == comm_label:

            return node

    return


def get_hop_nodes(g, node, hop_dist):

    node_list = [node]

    if hop_dist > 0:
        for nb in nx.neighbors(g, node):
            node_list.append(nb)

            if hop_dist > 1:
                for nb_nb in nx.neighbors(g, nb):
                    node_list.append(nb_nb)

                    if hop_dist > 2:
                        for nb_nb_nb in nx.neighbors(g, nb_nb):
                            node_list.append(nb_nb_nb)

                            if hop_dist > 3:
                                for nb_nb_nb_nb in nx.neighbors(g, nb_nb_nb):
                                    node_list.append(nb_nb_nb_nb)

    node_list = list(set(node_list))

    return node_list


#np.random.seed(0)
node = find_node_in_comm(g, node2comm, comm_label=1)
#node='810'
print(node)
nb_list = get_hop_nodes(g, node, hop_dist=3)
print(nb_list)
print([node2comm[node] for node in nb_list])

subg = nx.subgraph(g, nb_list)

node_colors = ['r', 'g', 'b', 'k', 'c', 'm']

plt.figure()
pos = nx.spring_layout(subg)
for k in range(K):
    com_nodes = [node for node in subg.nodes() if node2comm[node][0] == k]
    nx.draw_networkx_nodes(subg, pos, nodelist=com_nodes, node_color=node_colors[k], node_size=100, alpha=0.8)
nx.draw_networkx_edges(subg, pos, edgelist=subg.edges(), alpha=0.5)
plt.show()


#########################################################
comm2count = {k: 0 for k in range(K)}
for node in g.nodes():
    comm2count[node2comm[node][0]] += 1
print(comm2count)