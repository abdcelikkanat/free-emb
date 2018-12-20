import os, sys
import numpy as np
import pyximport
import networkx as nx

try:
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from get_neighbors import get_neighbors as gn
except ImportError:
    raise ImportError("'get_neighbors' module could not be imported!")


_BASE = os.path.dirname(__file__)
nxg_path = os.path.join(_BASE, "datasets", "karate.gml")


g = nx.read_gml(nxg_path)


nb_list = [[] for _ in range(nx.number_of_nodes(g))]
for node in g.nodes():
    for nb in nx.neighbors(g, node):
        nb_list[int(node)].append(int(nb))


gn.get_neighbors(nb_list=nb_list, n=g.number_of_nodes(), l=3)

