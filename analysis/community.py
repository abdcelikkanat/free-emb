import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

g = nx.read_gml("../datasets/citeseer_undirected.gml")

N = g.number_of_nodes()

###########################
exact_node2comm = nx.get_node_attributes(g, 'community')
for node in exact_node2comm:
    exact_node2comm[node] = [exact_node2comm[node]]


pred_node2comm = list(greedy_modularity_communities(g))

print(pred_node2comm)