import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate(sizes, matrix_p):

    cum_sizes = np.cumsum(sizes)

    g = nx.Graph()
    N = np.sum(sizes)
    comm_label = 0
    node2comm = {}
    for node in range(N):
        g.add_node(str(node))

        if node < cum_sizes[comm_label]:
            node2comm[str(node)] = comm_label
        else:
            comm_label += 1
            node2comm[str(node)] = comm_label

    for v in range(N):
        for u in range(v+1, N):
            if np.random.rand() < matrix_p[node2comm[str(v)]][node2comm[str(u)]]:
                g.add_edge(str(v), str(u))

    nx.set_node_attributes(G=g, name="community", values=node2comm)

    return g