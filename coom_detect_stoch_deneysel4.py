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
#g = nx.read_gml("./datasets/citeseer_undirected.gml")


'''
plt.figure()
nx.draw(g)
plt.show()
'''

sizes = [120, 180]
matrix_p = [[0.85, 0.45],
            [0.45, 0.75]]
#g = stoch_block_model.generate(sizes, matrix_p)


#g = nx.Graph()
#g.add_edges_from([[0,1], [1,2], [0,2], [3,4], [4,5], [3,5], [0,3]])


N = g.number_of_nodes()
K = 2
num_of_iters = 20  #0.82, 0.78, 0.001, 0.006, 0.78, 0.002, 0.047


def compute_prob(node, z, g, z_sum):
    #print(z_sum)
    p = np.zeros(shape=(K,), dtype=np.float)
    node_z = z[int(node), :]
    for nb in g.neighbors(node):
        p += z[int(nb), :]

    deg = g.degree(node)
    p = float(deg) - p

    z_inv = z_sum - node_z
    z_inv_sum = np.zeros(shape=z_sum.shape, dtype=np.float)
    for k in range(K):
        z_inv[k] += 1
        z_inv_sum[k] = np.sum(1.0 / z_inv)
        z_inv[k] -= 1

        p[k] = p[k] * z_inv_sum[k]

    p_sum = np.sum(p)
    p = p_sum - p
    #print(p, node)
    p = p / np.sum(p)

    return p


def compute_beta(b0, beta, g):

    b = np.exp(b0 - 1.0)

    return np.add(b, beta)

# Initialize labels
found = np.zeros(shape=(N,), dtype=np.int)
expId = 1
while(np.sum(found)) == N or np.sum(found) == 0 and expId < 2:

    #beta = np.random.dirichlet(alpha=[0.5, 0.5], size=(N, ))
    #print("Init: ", np.argmax(beta, axis=1))

    z = np.zeros(shape=(N, K), dtype=np.float)
    for i in range(N):
        z[i, np.random.choice(K, size=1)] = 1

    z_sum = np.sum(z, axis=0)

    print("Exp Id {}".format(expId))
    expId += 1
    for iter in range(num_of_iters):
        for node in g.nodes():
            p = compute_prob(node, z, g, z_sum)
            current_z = z[int(node), :]
            z_sum -= current_z
            #print("as: ", current_z)
            z[int(node), :] -= current_z
            k = np.random.choice(K, size=1, p=p)
            z[int(node), k] = 1
            z_sum += z[int(node), ]

        print(np.argmax(z, axis=1))

    found = np.argmax(z, axis=1)
print("last: ", np.argmax(z, axis=1))


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


'''


gt_node2comm = nx.get_node_attributes(g, 'community')
correct_labels = [gt_node2comm[str(node)] for node in range(N)]
pred_labels = [found[node] for node in range(N)]

nmi = normalized_mutual_info_score(correct_labels, pred_labels)

print("NMI: {}".format(nmi))



found2 = community.best_partition(graph=g)
pred_labels = [found2[str(node)] for node in range(N)]
nmi = normalized_mutual_info_score(correct_labels, pred_labels)

print("Louvain NMI: {}".format(nmi))

