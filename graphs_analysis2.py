import os
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_min_max_avg_degree(g):
    min_degree = g.number_of_nodes()
    max_degree = 0
    avg_degree = 0.0

    for node in g.nodes():
        current_degree = nx.degree(g, node)
        if current_degree > max_degree:
            max_degree = current_degree
        if current_degree < min_degree:
            min_degree = current_degree

        avg_degree += current_degree

    return max_degree, min_degree, avg_degree/g.number_of_nodes()

def get_degree_sequence(g):

    degree_seq = []
    for node in g.nodes():
        degree_seq.append(nx.degree(g, node))

    return sorted(degree_seq)[::-1]

# Buradan basliyor

graph_name = "blogcatalog.gml"
#graph_name = "cora_undirected.gml"
g = nx.read_gml(os.path.join("./datasets", graph_name))



N = g.number_of_nodes()
E = g.number_of_edges()
nodes = list(g.nodes())

print("Number of nodes: {}".format(N))
print("Number of edges: {}".format(E))

max_deg, min_deg, avg_deg = get_min_max_avg_degree(g)
print("Min degree: {}".format(min_deg))
print("Max degree: {}".format(max_deg))
print("Avg degree: {}".format(avg_deg))

deg_seq = get_degree_sequence(g)
plt.figure()
plt.hist(deg_seq, range=(min(deg_seq), max(deg_seq)), bins=[b for b in np.arange(min(deg_seq)-0.5, max(deg_seq)+1.5, 1)])
#plt.show()

# Computes community dictionary and finds num_of_labels
num_of_labels = 0
community = nx.get_node_attributes(g, 'community')

for node in nodes:
    communities = community[node]
    if type(communities) is not list:
        community[node] = [communities]

    if max(community[node])+1 > num_of_labels:
        num_of_labels = max(community[node])+1

    #print(community[node])
### end ####

print(community)

# Find the node having maximum degree

degree_list = sorted(g.degree(), key=lambda x: x[1], reverse=True)

max_node = degree_list[83][0]
print(max_node)

l = [nx.degree(g, nb) for nb in nx.neighbors(g, max_node)]
print(l)
print(max(l))
print(min(l))
print(np.average(l))
#s = [( nb, len(list(nx.common_neighbors(g, max_node, nb)))/float(min(g.degree(max_node), g.degree(nb))) ) for nb in nx.neighbors(g, max_node)]

s = [( nb, len(list(nx.common_neighbors(g, max_node, nb))) ) for nb in nx.neighbors(g, max_node)]

s = [ (nb, 1) for nb in nx.neighbors(g, max_node)]

s = sorted(s, key=lambda s: s[1], reverse=True)


print(s)

print("--------")
print(community[max_node])
com_counts = np.zeros(shape=(num_of_labels, ), dtype=np.int)
for p in s[:512]:
    nb_coms = community[p[0]]
    for x in nb_coms:
        com_counts[int(x)] += 1

print(com_counts)


