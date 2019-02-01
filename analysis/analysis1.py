import numpy as np
import networkx as nx
from utils import *

g = nx.read_gml("../datasets/blogcatalog.gml")

node2comm, K = getCommunities(g)

for node in g.nodes():
    nb_coms = np.zeros(shape=(K, ), dtype=np.int)
    node_coms = np.zeros(shape=(K, ), dtype=np.int)

    node_coms[node2comm[node]] += 1
    for nb in nx.neighbors(g, node):
        nb_coms[node2comm[nb]] += 1

    node_k = np.sum(node_coms)

    nb_coms.argsort()[-3:]

    print(nb_coms)