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

#graph_name = "blogcatalog.gml"
graph_name = "cora_undirected.gml"
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


# Computes labels in neighbors
total_estimation_success = 0
for node in nodes:
    node_true_label_count = len(community[node])

    label_counts = np.zeros(shape=(num_of_labels, ), dtype=np.int)

    for nb in nx.neighbors(g, node):
        nb_labels = community[nb]
        for l in nb_labels:
            label_counts[l] += 1


        for nb_nb in nx.neighbors(g, nb):
            if nb_nb != node:
                nb_nb_labels = community[nb_nb]
                for l in nb_nb_labels:
                    label_counts[l] += 1


    most_freq_labels = np.argsort(label_counts)[::-1][:node_true_label_count]

    node_estimation_success = 0.0
    for l in most_freq_labels:
        if l in community[node]:
            node_estimation_success += 1.0
    node_estimation_success = node_estimation_success / node_true_label_count

    total_estimation_success += node_estimation_success

total_estimation_success = total_estimation_success / float(N)

print("Success: %{}".format(total_estimation_success*100))


'''   
# Computes labels in triangles
total_estimation_success = 0
for node in nodes:
    node_true_label_count = len(community[node])

    label_counts = np.zeros(shape=(num_of_labels, ), dtype=np.int)

    for nb in nx.neighbors(g, node):
        if nb != node:
            for nb_nb in nx.neighbors(g, nb):
                if nb_nb != node and nb_nb != nb:

                    for nb_nb_nb in nx.neighbors(g, nb_nb):
                        if nb_nb_nb == node:

                            nb_labels = community[nb]
                            for l in nb_labels:
                                label_counts[l] += 1

                            nb_nb_labels = community[nb_nb]
                            for l in nb_nb_labels:
                                label_counts[l] += 1

    most_freq_labels = np.argsort(label_counts)[::-1][:node_true_label_count]

    node_estimation_success = 0.0
    for l in most_freq_labels:
        if l in community[node]:
            node_estimation_success += 1.0
    node_estimation_success = node_estimation_success / node_true_label_count

    total_estimation_success += node_estimation_success

total_estimation_success = total_estimation_success / float(N)

print("Success: %{}".format(total_estimation_success*100))
'''


# Computes labels fixed neighbors
total_estimation_success = 0
for node in nodes:
    node_true_label_count = len(community[node])

    label_counts = np.zeros(shape=(num_of_labels, ), dtype=np.int)

    nb_list = list(nx.neighbors(g, node))
    nb_deg_list = [nx.degree(g, nb) for nb in nb_list]

    sorted_nb_deg_list = [x for x, _ in sorted(zip(nb_list, nb_deg_list), key=lambda pair: pair[1])]
    print("List {} sorted: {}".format(nb_list, sorted_nb_deg_list))

print("Success: %{}".format(total_estimation_success*100))