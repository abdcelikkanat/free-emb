import numpy as np
import networkx as nx


def generate_walks(g, spl, l, n):

    walks = []

    for _ in range(n):

        for node in g.nodes():

            walk = [node]
            for _ in range(1, l):
                current = walk[-1]
                nb_list = list(nx.neighbors(g, current))
                next = np.random.choice(a=nb_list, size=1)[0]
                walk.append(next)
            walks.append(walk)

    return walks


g = nx.read_gml("../datasets/citeseer_undirected.gml")
N = g.number_of_nodes()


spl = nx.shortest_path_length(g)
walks = generate_walks(g, spl, l=10, n=80)

with open('./citeseer_deepwalk_test.corpus', 'w') as f:
    for walk in walks:
        f.write("{}\n".format(" ".join([str(w) for w in walk])))
