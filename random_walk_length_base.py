import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


g = nx.read_gml("../datasets/citeseer_undirected.gml")
N = g.number_of_nodes()

########################################################
node2comm = nx.get_node_attributes(g, 'community')
for node in node2comm:
    if type(node2comm[node]) is int:
        node2comm[node] = [node2comm[node]]
########################################################

def perform_a_walk(g, node, spl, n, l):

    walks = []
    for _ in range(n):
        walk = [node]

        for _ in range(l):
            current_node = walk[-1]
            nb_list = list(nx.neighbors(g, current_node))
            nb_prob = np.asarray([1.0 if spl[node][nb] == 0 else (1.0 / spl[node][nb]**2) for nb in nb_list])
            nb_prob = nb_prob / float(np.sum(nb_prob))
            next_node = np.random.choice(a=nb_list, p=nb_prob, size=1)[0]
            walk.append(next_node)

        walks.append(walk)


    return walks


def count_nodes_in_walks(walks):

    counts = {}

    for walk in walks:
        for node in walk[1:]:
            if node in counts:
                counts[node] += 1
            else:
                counts[node] = 1

    return counts

'''
node = '732'
spl = nx.shortest_path_length(g)
walks = perform_a_walk(g, node, spl, n=500, l=2)
counts = count_nodes_in_walks(walks=walks)

for key, value in sorted(counts.iteritems(), key=lambda (key, value): (value, key)):
    print "%s: %s === %s, distance: %s" % (key, value, node2comm[key], spl[node][key])


for nb in nx.neighbors(g, node):
    print("Node: {} = {}, degree: {}".format(nb, node2comm[nb], nx.degree(g, nb)))



subg = g.subgraph(counts.keys())

#plt.figure()
#nx.draw(subg, with_labels=True)
#plt.show()

'''

true_counter = 0.0
total_counter = 0.0

ll = 3

spl = nx.shortest_path_length(g)
for node in g.nodes():
    true_label = node2comm[node]

    walks = perform_a_walk(g, node, spl, n=100, l=5)
    counts = count_nodes_in_walks(walks=walks)
    for key, value in sorted(counts.iteritems(), key=lambda (key, value): (value, key))[-ll-1:-1]:
        if key != node and node2comm[node] == node2comm[key]:
            true_counter += 1.0

total_counter += float(ll)*N

print(100*true_counter/total_counter)