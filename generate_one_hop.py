import numpy as np
import networkx as nx


def generate_walks_onehop(g, spl, l, n):

    walks = []

    for _ in range(n):

        for node in g.nodes():

            walk = [node]
            for _ in range(1, l):
                current = node
                nb_list = list(nx.neighbors(g, current))
                next = np.random.choice(a=nb_list, size=1)[0]
                walk.append(next)
            walks.append(walk)

    return walks


def generate_walks_twohop(g, spl, l, n):

    twohops = {node: [node] for node in g.nodes()}
    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            for nb_nb in nx.neighbors(g, nb):
                if spl[node][nb_nb] == 2 and nb_nb not in twohops[node]:
                    twohops[node].append(nb_nb)


    walks = []

    for _ in range(n):

        for node in g.nodes():

            walk = [node]
            for _ in range(1, l):
                #current = node
                #nb_list = list(nx.neighbors(g, current))
                next = np.random.choice(a=twohops[node], size=1)[0]
                walk.append(next)
            walks.append(walk)

    return walks


def generate_walks_one_two_hop(g, spl, l, n):
    onehops = {node: [] for node in g.nodes()}
    twohops = {node: [node] for node in g.nodes()}
    for node in g.nodes():
        for nb in nx.neighbors(g, node):
            onehops[node].append(nb)
            for nb_nb in nx.neighbors(g, nb):
                if spl[node][nb_nb] == 2 and nb_nb not in twohops[node]:
                    twohops[node].append(nb_nb)


    walks = []

    for _ in range(n):

        for node in g.nodes():

            nb_list = onehops[node] + twohops[node]

            walk = [node]
            for _ in range(1, l):
                #current = node
                #nb_list = list(nx.neighbors(g, current))
                next = np.random.choice(a=nb_list, size=1)[0]
                walk.append(next)
            walks.append(walk)

    return walks


g = nx.read_gml("../datasets/citeseer_undirected.gml")
N = g.number_of_nodes()


spl = nx.shortest_path_length(g)
#walks = generate_walks_onehop(g, spl, l=10, n=80)
walks = generate_walks_one_two_hop(g, spl, l=10, n=80)

with open('./citeseer_test.corpus', 'w') as f:
    for walk in walks:
        f.write("{}\n".format(" ".join([str(w) for w in walk])))
