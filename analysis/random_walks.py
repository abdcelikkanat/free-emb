import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



g = nx.read_gml("../datasets/blogcatalog.gml")

spl = dict(nx.shortest_path_length(g))
#print(list(nx.neighbors(g, n='0')))



L = 80
len_sum = {node: 0.0 for node in g.nodes()}

spl_counter = np.zeros(shape=(L, ), dtype=np.int)

for _ in range(10):
    for node in g.nodes():
        path = [node]
        for l in range(2, L+1):
            current = path[-1]
            next = np.random.choice(a=list(nx.neighbors(g, current)))
            path.append(next)
            len_sum[node] += spl[node][next]

            spl_counter[spl[node][next]] += 1

        len_sum[node] /= (L-1.0)


avg_len = sum(len_sum.values()) / (g.number_of_nodes())



for node in g.nodes():
    print(len_sum[node])

print("---------------")

print(avg_len)
print(np.mean(a=[nx.degree(g, node) for node in g.nodes()]))

#plt.figure()
#plt.hist(avg_len)
#plt.show()

print(spl_counter/float(np.sum(spl_counter))*100.0)
print(np.sum(spl_counter[0:3])/float(np.sum(spl_counter))*100.0)