import networkx as nx
import numpy as np

def sbm(sizes, prob_matrix):

    def get_comm_label(node):
        print("test")

    g = nx.Graph()
    N = np.sum(sizes)

    last_node_labels = np.cumsum(sizes)

    for source in range(N):
        for target in range(source+1, N):
            pass

    tst()


p_matrix = np.asarray([[0.7, 0.3], [0.3, 0.8]])
g = sbm(sizes=[4, 5], prob_matrix=p_matrix)