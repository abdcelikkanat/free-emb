import community
import networkx as nx
import numpy as np


g = nx.read_gml("./datasets/blogcatalog.gml")
#g = nx.read_gml("./datasets/karate.gml")

node2comm = community.best_partition(g)
num_of_comms = max(node2comm.values()) + 1

print(num_of_comms)

counts = np.zeros(shape=(num_of_comms), dtype=np.float)

for node in node2comm:
    counts[int(node2comm[node])] += 1

print(np.max(counts))