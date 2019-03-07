import numpy as np
import networkx as nx


def generate_walks(g, spl, l, n):

    walks = []

    for _ in range(n):

        for node in g.nodes():

            walk = [node]

            while len(walk) < l:
                current = walk[-1]
                nb_list = list(nx.neighbors(g, current))
                nb_prob = np.asarray([1.0 if spl[node][nb] == 0 else (1.0 / spl[node][nb] ** -2.0) for nb in nb_list])
                nb_prob = nb_prob / float(np.sum(nb_prob))
                next = np.random.choice(a=nb_list, size=1, p=nb_prob)[0]

                '''
                if walk[-1] == node and nx.degree(g, node) > 2:
                    walk[-1] = next
                else:
                    walk.append(next)
                    ll += 1
                '''
                walk.append(next)

            walks.append(walk)

    return walks


g = nx.read_gml("../datasets/citeseer_undirected.gml")
N = g.number_of_nodes()


spl = nx.shortest_path_length(g)
walks = generate_walks(g, spl, l=10, n=80)

with open('./citeseer_test.corpus', 'w') as f:
    for walk in walks:
        f.write("{}\n".format(" ".join([str(w) for w in walk])))
