import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
import stoch_block_model
import community


g = nx.Graph()
g.add_edges_from([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                  [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7],
                  [3, 4] ])

#g = nx.read_gml("./datasets/karate.gml")
g = nx.read_gml("./datasets/citeseer_undirected.gml")


'''
plt.figure()
nx.draw(g)
plt.show()
'''

sizes = [12, 18]
matrix_p = [[0.85, 0.15],
            [0.15, 0.75]]
g = stoch_block_model.generate(sizes, matrix_p)


#g = nx.Graph()
#g.add_edges_from([[0,1], [1,2], [0,2], [3,4], [4,5], [3,5], [0,3]])


N = g.number_of_nodes()
K = 2
num_of_iters = 250  #0.82, 0.78, 0.001, 0.006, 0.78, 0.002, 0.047


def compute_prob(node, beta, g):
    p = np.zeros(shape=(K,), dtype=np.float)
    for nb in g.neighbors(node):
        p += beta[int(nb), :]
    p = np.exp(p)
    p = p / np.sum(p)

    return p


def compute_beta(node, b0, beta, g):

    b = np.exp(b0 - 1.0)

    return np.add(b, beta)

# Initialize labels
beta = np.random.dirichlet(alpha=[0.5, 0.5], size=(N, ))
print("Init: ", np.argmax(beta, axis=1))

for iter in range(num_of_iters):
    for node in g.nodes():
        beta[int(node), :] = compute_prob(node, beta, g)


found = np.argmax(beta, axis=1)
print("last: ", np.argmax(beta, axis=1))


pos = nx.spring_layout(g)  # positions for all nodes


# nodes
nx.draw_networkx_nodes(g, pos,
                       nodelist=[node  for node in g.nodes() if found[int(node)]==1],
                       node_color='r',
                       node_size=100,
                       alpha=0.8)

nx.draw_networkx_nodes(g, pos,
                       nodelist=[node for node in g.nodes() if found[int(node)]==0],
                       node_color='b',
                       node_size=100,
                       alpha=0.8)

nx.draw_networkx_edges(g, pos,width=1.0, alpha=0.5)

'''
plt.show()




gt_node2comm = nx.get_node_attributes(g, 'community')
correct_labels = [gt_node2comm[str(node)] for node in range(N)]
pred_labels = [found[node] for node in range(N)]

nmi = normalized_mutual_info_score(correct_labels, pred_labels)

print("NMI: {}".format(nmi))



found2 = community.best_partition(graph=g)
pred_labels = [found2[str(node)] for node in range(N)]
nmi = normalized_mutual_info_score(correct_labels, pred_labels)

print("Louvain NMI: {}".format(nmi))

'''