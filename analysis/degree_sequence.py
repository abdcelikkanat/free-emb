import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils import *


g = nx.read_gml("../datasets/blogcatalog.gml")

N = g.number_of_nodes()


d = [nx.degree(g, node) for node in g.nodes()]

print(np.arange(min(d)-0.5, max(d)+1.5, 1.0))


plt.figure()
plt.hist(d, bins=[b for b in np.arange(min(d)-0.5, max(d)+1.5, 1.0)])
plt.show()