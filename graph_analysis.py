import os
import sys
import networkx as nx
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


graph_name = "blogcatalog.gml"
g = nx.read_gml(os.path.join("./datasets", graph_name))



N = g.number_of_nodes()
E = g.number_of_edges()
nodes = list(g.nodes())

print("Number of nodes: {}".format(N))
print("Number of edges: {}".format(E))

min_deg, max_deg, avg_deg = get_min_max_avg_degree(g)
print("Min degree: {}".format(min_deg))
print("Max degree: {}".format(max_deg))
print("Avg degree: {}".format(avg_deg))

deg_seq = get_degree_sequence(g)
#plt.figure()
#plt.hist(deg_seq)
#plt.show()



community = nx.get_node_attributes(g, 'community')

for node in nodes:
    communities = community[node]
    if type(communities) is not list:
        community[node] = [communities]

    #print(community[node])


