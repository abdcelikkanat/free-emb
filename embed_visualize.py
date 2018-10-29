import numpy as np
import matplotlib.pyplot as plt

embed_file = "./outputs/citeseer_node2vec.embedding"

with open(embed_file, 'r') as f:
    f.readline() # skip first line

    node2vec = {}
    for line in f.readlines():
        node, x, y = line.strip().split()
        node2vec[node] = [float(x), float(y)]


x_coor = [node2vec[node][0] for node in node2vec]
y_coor = [node2vec[node][1] for node in node2vec]

plt.figure()
plt.plot(x_coor, y_coor, 'x')
plt.show()