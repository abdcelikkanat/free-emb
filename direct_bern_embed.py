import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class myBigclam:
    def __init__(self, nxg=None, dim_size=2):

        self._graph = None
        self._num_of_nodes = None
        self._edgelist = None
        self._F = None
        self._current_f_sum = 0.0
        self._dim_size = dim_size
        self._nb_list = []

        #self._read_gml(nxg_path)
        if nxg is not None:
            self._graph = nxg
            self._num_of_nodes = self._graph.number_of_nodes()
            self._nx2list()
            self._get_nb_strategy1()

    def _read_gml(self, nxg_path):
        self._graph = nx.read_gml(nxg_path)
        self._num_of_nodes = self._graph.number_of_nodes()
        self._nx2list()

    def _nx2list(self):
        self._nodelist = [node for node in range(self._graph.number_of_nodes())]
        self._edgelist = [[] for _ in range(self._graph.number_of_nodes())]

        for node in self._graph.nodes():
            for nb in nx.neighbors(self._graph, node):
                self._edgelist[int(node)].append(int(nb))

        return self._nodelist

    def _get_nb_strategy1(self):
        self._nb_list = []

        for v in range(self._num_of_nodes):
            self._nb_list.append([])

            for nb in self._edgelist[v]:
                if nb != v:
                    self._nb_list[v].append(nb)
                    for nb_nb in self._edgelist[nb]:
                        if nb_nb != v:
                            self._nb_list[v].append(nb_nb)

        #self._nb_list = np.unique(self._nb_list)


    def sigmoid(self, z):

        return 1.0 / ( 1.0 + np.exp(-z) )

    def compute_gradient(self, v):

        grad = 0.0
        for u in range(self._num_of_nodes):

            if u != v:

                label = 0.0
                if u in self._nb_list[v]:
                    label = 1.0

                z = np.dot(self._F[v, :], self._F[u, :])
                grad += ( label - self.sigmoid(z) ) * self._F[u, :]

        return grad

    def save_f(self, filename):

        with open(filename, 'w') as f:
            f.write("{} {}\n".format(self._num_of_nodes, self._dim_size))
            for v in range(self._num_of_nodes):
                f.write("{} {}\n".format(str(v), " ".join([str(value) for value in F[v, :]])))


    def run(self, starting_alpha, num_of_iterations):

        self._F = np.random.rand(self._num_of_nodes, self._dim_size)
        self._current_f_sum = np.sum(self._F, 0)

        alpha = starting_alpha

        for iter in range(num_of_iterations):

            for v in range(self._num_of_nodes):
                temp = alpha * self.compute_gradient(v)
                self._F[v, :] += temp




            if iter % 10 == 0:
                log_score = 0.0
                for v in range(self._num_of_nodes):
                    for u in range(v+1, self._num_of_nodes):
                        sig = self.sigmoid(np.dot(self._F[v, :], self._F[u, :]))

                        if u in self._nb_list[v]:
                            log_score += np.log(sig)
                        else:
                            log_score += np.log( 1.0 - sig )

                print("Iter: {} Total log score: {}".format(iter, log_score))

        #epsilon = (2.0 * self._graph.number_of_edges()) / ( self._num_of_nodes * (self._num_of_nodes-1) )

        return self._F

    def plot(self, x):

        plt.figure()

        plt.plot(x[:, 0], x[:, 1], '.')

        plt.show()

path = "./datasets/citeseer_undirected.gml"
#path = "./datasets/karate.gml"
g = nx.read_gml(path)

#g = nx.Graph()
#g.add_edges_from([[0,1], [0,2], [1,2], [0,3], [1,3], [2,3],  [4,5], [4,6], [4,7], [5,6],  [5,7], [6,7], [3,4]])

#nx.draw(g)
#plt.show()


bg = myBigclam(nxg=g, dim_size=128)
F = bg.run(starting_alpha=0.001, num_of_iterations=1000)
"""
print(F)
M = np.zeros(shape=(g.number_of_nodes(), g.number_of_nodes()), dtype=np.float)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        M[i, j] = bg.sigmoid(np.dot(F[i, :], F[j, :]))

print(M)

bg.plot(x=F)

"""

bg.save_f(filename="./outputs/citeseer_test.embedding")