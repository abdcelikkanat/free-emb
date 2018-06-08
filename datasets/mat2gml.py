import networkx as nx
import numpy as np
import scipy.io as sio


dataset_name = "blogcatalog.mat"
network_name = "network"
group_name = "group"

# read mat file
mat_dict = sio.loadmat(dataset_name)

N = mat_dict[network_name].shape[1]
K = mat_dict[group_name].shape[1]

print("Number of nodes: {}".format(N))
print("Number of classes: {}".format(K))

g = nx.Graph()

network_mat = mat_dict[network_name].tocoo()
cx = network_mat.tocoo()
for i, j, val in zip(cx.row, cx.col, cx.data):
    if val > 0:
        g.add_edge(str(i), str(j))

# Set the cluster labels
for node in g.nodes():
    g.node[node]["clusters"] = []

group_mat = mat_dict[group_name].tocoo()
ck = group_mat.tocoo()
for i, k, val in zip(ck.row, ck.col, ck.data):
    if val > 0:
        g.node[str(i)]['clusters'].append(str(k))



nx.write_gml(g, "./blogcatalog.gml")
