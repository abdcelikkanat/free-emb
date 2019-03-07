import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_gml("../datasets/citeseer_undirected.gml")
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


def perform_random_walk(g, node, max_len):

    walk = [node]

    for i in range(1, max_len):
        current_node = walk[-1]

        nb_list = list(nx.neighbors(g, current_node))

        next_node = np.random.choice(a=nb_list, size=1)[0]

        walk.append(next_node)

    return walk


def count_nodes_in_a_walk(g, walk):
    N = g.number_of_nodes()
    counts = np.zeros(shape=(N, ), dtype=np.int)

    for i in range(len(walk)):
        counts[int(walk[i])] += 1

    return counts

g = nx.Graph()

for i in range(3):
    for j in range(i+1, 3):
        g.add_edge(str(i), str(j))

for i in range(3, 10):
    for j in range(i+1, 10):
        g.add_edge(str(i), str(j))

g.add_edge(str(2), str(3))

#plt.figure()
#nx.draw(g)
#plt.show()


walk = perform_random_walk(g, node='0', max_len=1000000)
counts = count_nodes_in_a_walk(g, walk)

print(counts/float(np.sum(counts)))
deg_seq = np.asarray([nx.degree(g, str(i)) for i in range(10)])
print(deg_seq / float(np.sum(deg_seq)) )