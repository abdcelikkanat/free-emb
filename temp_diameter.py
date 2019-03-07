import networkx as nx

edges = [[0,1], [1, 2], [0,2], [3, 2], [3, 4], [4, 5], [3, 5]]


g = nx.Graph()
g.add_edges_from(edges)

diam = nx.diameter(g)
print("Diameter: {}".format(diam))