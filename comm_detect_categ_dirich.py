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

sizes = [120, 180]
matrix_p = [[0.85, 0.25],
            [0.25, 0.75]]
g = stoch_block_model.generate(sizes, matrix_p)

N = g.number_of_nodes()
K = 2
num_of_iters = 5000 #0.82, 0.78, 0.001, 0.006, 0.78, 0.002, 0.047


# Initialize alpha_tilde
alpha_tilde = np.ones(shape=(N, K), dtype=np.float) / float(K)
alpha_tilde = np.random.dirichlet([0.6, 0.4], size=(N))

# Initialize labels
#node_labels = [np.random.choice(K, p=alpha_tilde[node, :], size=1)[0] for node in range(N)]

alpha = np.zeros(shape=(N, K), dtype=np.float)


'''
for iter in range(num_of_iters):
    node = np.random.choice(N)
    nb_labels = np.asarray([node_labels[nb] for nb in g.neighbors(node)])
    nb_counts = np.asarray([np.sum(nb_labels == k) for k in range(K)])  #this line can be more efficient
    alpha[node, :] = alpha_tilde[node, :] + nb_counts
    p = np.random.dirichlet(alpha[node, :])
    node_labels[node] = np.random.choice(K, p=p, size=1)[0]

'''

'''
for iter in range(num_of_iters):
    print(node_labels)
    for node in range(N):
        nb_labels = np.asarray([node_labels[nb] for nb in g.neighbors(node)])
        nb_counts = np.asarray([np.sum(nb_labels == k) for k in range(K)])  #this line can be more efficient
        p = nb_counts / float(np.sum(nb_counts))
        print(p)
        node_labels[node] = np.random.choice(K, p=p, size=1)[0]

print(alpha_tilde)
print(np.argmax(alpha_tilde, axis=1))
print(alpha)

'''


node2comm = np.zeros(shape=(N, K), dtype=np.int)
for node in range(N):
    node2comm[node, np.random.choice(K)] = 1


alpha1 = 0.005
alpha2 = 0.0002


print(np.argmax(node2comm, axis=1))

for iter in range(num_of_iters):
    node = str(np.random.choice(N))
    prob = np.asarray([np.exp(alpha1*np.sum([node2comm[int(nb), z] for nb in nx.neighbors(g, node)]) +
                              alpha2*np.sum([node2comm[int(nb_nb), z] for nb_nb in nx.neighbors(g, nb) for nb in nx.neighbors(g, node) if nb_nb in nx.neighbors(g, node)]
                                            )) for z in range(K)])
    prob = np.exp(np.log(prob) - np.log(np.sum(prob)))
    k = np.random.choice(K, p=prob)
    node2comm[int(node), :] = np.zeros(shape=(K, ), dtype=np.int)
    node2comm[int(node), k] = 1


found = np.argmax(node2comm, axis=1)
print(found)

'''
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