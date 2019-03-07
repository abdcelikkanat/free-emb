import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_gml("../datasets/citeseer_undirected.gml")

N = g.number_of_nodes()

###########################
node2comm = nx.get_node_attributes(g, 'community')
for node in node2comm:
    node2comm[node] = [node2comm[node]]
############################

############################
K = 0
for node in node2comm:
    m = max(node2comm[node])
    if K < m:
        K = m
K += 1
############################


#for node in node2comm:
#    print(node2comm[node])



def dissolve(K, node2comm, g, node):

    bin = np.zeros(shape=(K, ), dtype=np.int)

    for nb in nx.neighbors(g, node):
        bin[node2comm[nb][0]] += 1

    return bin


deg_seq = np.argsort([nx.degree(g, str(node)) for node in range(N)])[::-1][:10]
print(deg_seq)


counter = 0
for node in g.nodes():
    b = np.argmax(dissolve(K, node2comm, g, node))
    jac = list(nx.jaccard_coefficient(g, [(node, str(b))]))
    if node2comm[node][0] != b:
        if nx.degree(g,node) != 1:
            #print("node: {}, jaccard: {}".format(node, jac[0][2]))
            3+3
    else:
        if jac[0][2] == 0.0:
            if nx.degree(g, str(b)) > 1:
                print("BURDA: {} {}".format(b, nx.degree(g, str(b))))
                counter += 1

print("Counter: {}".format(counter))

node = "0"
b = dissolve(K, node2comm, g, node)
print(b)
print("Node: {} Nb: {}".format(node2comm[node][0], np.argmax(b)))
print("Node degree: {}".format(nx.degree(g, node)))



def partition(g):
    node2comm = {node: [] for node in g.nodes()}

    for node in g.nodes():

        for nb in nx.neighbors(g, node):
            3+3

    return node2comm