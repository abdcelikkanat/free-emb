import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class myBigclam:
    def __init__(self, nxg_path, num_of_comm):

        self._graph = None
        self._num_of_nodes = None
        self._edgelist = None
        self._F = None
        self._current_f_sum = 0.0
        self._num_of_comm = num_of_comm

        self._read_gml(nxg_path)
        self.nx2list()

    def _read_gml(self, nxg_path):
        self._graph = nx.read_gml(nxg_path)
        self._num_of_nodes = self._graph.number_of_nodes()

    def nx2list(self):
        self._nodelist = [node for node in range(self._graph.number_of_nodes())]
        self._edgelist = [[] for _ in range(self._graph.number_of_nodes())]

        for node in self._graph.nodes():
            for nb in nx.neighbors(self._graph, node):
                self._edgelist[int(node)].append(int(nb))

        return self._nodelist

    def sigmoid(self, z):

        return 1.0 / ( 1.0 + np.exp(-z) )

    def compute_gradient(self, u):

        grad = 0.0
        v_f_sum = 0.0
        for v in self._edgelist[u]:
            z = np.dot(self._F[u, :], self._F[v, :])
            grad += self._F[v, :]*self.sigmoid(-z)
            v_f_sum = self._F[v, :]

        partial_sum = self._current_f_sum - self._F[u, :] - v_f_sum

        ########
        partial_sum += -self._F[u, :]
        #########

        return grad - partial_sum

    def _get_comm_assignment(self, epsilon=1e-8):

        node2comm = {node:[] for node in self._nodelist}
        com2node = {c: [] for c in range(self._num_of_comm)}
        for node in self._nodelist:
            for c in range(self._num_of_comm):
                if self._F[node, c] > epsilon:
                    node2comm[node].append(c)
                    com2node[c].append(node)


        return node2comm, com2node

    def run(self, starting_alpha, epsilon, num_of_iterations):

        self._F = np.random.rand(self._num_of_nodes, self._num_of_comm)
        self._current_f_sum = np.sum(self._F, 0)

        alpha = starting_alpha

        for iter in range(num_of_iterations):

            for u in self._nodelist:
                temp = alpha * self.compute_gradient(u)
                self._F[u, :] += temp
                #self._current_f_sum += temp

                self._current_f_sum = np.sum(self._F, 0)

                """
                for c in range(self._num_of_comm):
                    if self._F[u, c] < 0.0:
                        self._current_f_sum[c] -= temp[c]
                        self._F[u, c] = 0.0
                """


            log_score = 0.0
            for u in range(self._num_of_nodes):
                for v in range(u, self._num_of_nodes):

                    if v in self._edgelist[u]:

                        term = 1.0 - np.exp(-np.dot(self._F[u, :], self._F[v, :]))
                        if term < 1e-15:
                            log_score += -15
                        else:
                            log_score += np.log(term)

                    else:

                        log_score += - np.dot(self._F[u, :], self._F[v, :])


            print("Iter: {} Total log score: {}".format(iter, log_score))

        #epsilon = (2.0 * self._graph.number_of_edges()) / ( self._num_of_nodes * (self._num_of_nodes-1) )
        print(epsilon)

        return self._get_comm_assignment(epsilon=epsilon)

    def plot(self, node2com):

        newg = nx.Graph()
        for node in range(self._graph.number_of_nodes()):
            for nb in self._edgelist[node]:
                if node <= nb:
                    newg.add_edge(node, nb)

        pos = nx.spring_layout(newg)
        colors = ['b', 'r', 'g', 'w']

        plt.figure()

        for node in range(self._num_of_nodes):
            if len(node2com[node])> 1:
                color = colors[-2]
            elif len(node2com[node]) == 1:
                color = colors[node2com[node][0]]
            else:
                color = colors[-1]
            nx.draw_networkx_nodes(newg, pos,
                                   nodelist=[node],
                                   node_color=color,
                                   node_size=100,
                                   alpha=0.8)

        # edges
        nx.draw_networkx_edges(newg, pos, width=1.0, alpha=0.5)

        plt.show()

path = "./datasets/karate.gml"
bg = myBigclam(nxg_path=path, num_of_comm=2)
n2c, c2n = bg.run(starting_alpha=0.001, epsilon=1e-8, num_of_iterations=1000)
print(n2c)
print(c2n)

bg.plot(node2com=n2c)