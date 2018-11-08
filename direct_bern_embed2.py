import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import time

uni = None

class Vocab:

    def __init__(self, N, freq):
        self.nodes = [i for i in range(N)]
        self.freq = freq

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab, power = 0.75):
        self.vocab_size = len(vocab)

        norm = sum([math.pow(vocab.freq[t], power) for t in vocab])  # Normalizing constant

        # table_size = 1e8 # Length of the unigram table
        table_size = np.uint32(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(vocab.freq[unigram], power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


class myBigclam:
    def __init__(self, nxg=None, dim_size=2):

        self._graph = None
        self._num_of_nodes = None
        self._edgelist = None
        #self._F = None
        self._F0 = None
        self._F1 = None
        self._current_f_sum = 0.0
        self._dim_size = dim_size
        self._nb_list = []
        self.neg_count = 0

        #self._read_gml(nxg_path)
        if nxg is not None:
            self._graph = nxg
            self._num_of_nodes = self._graph.number_of_nodes()
            self._nx2list()


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

        #self._nb_list = np.unique(self._nb_list)

    def _get_nb_strategy2(self):
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


    def _get_nb_strategy3(self):
        self._nb_list = []

        for v in range(self._num_of_nodes):
            self._nb_list.append([])

            for nb in self._edgelist[v]:
                if nb != v:
                    self._nb_list[v].append(nb)
                    for nb_nb in self._edgelist[nb]:
                        if nb_nb != v:
                            self._nb_list[v].append(nb_nb)
                            for nb_nb_nb in self._edgelist[nb_nb]:
                                if nb_nb_nb != v:
                                    self._nb_list[v].append(nb_nb_nb)
                else:
                    if len(self._edgelist[v]) == 1:
                        self._nb_list[v].append(v)

    def _get_nb_strategy4(self):
        self._nb_list = []

        for v in range(self._num_of_nodes):
            self._nb_list.append([])

            for nb in self._edgelist[v]:
                if nb != v:
                    self._nb_list[v].append(nb)
                    for nb_nb in self._edgelist[nb]:
                        if nb_nb != v:
                            self._nb_list[v].append(nb_nb)
                            for nb_nb_nb in self._edgelist[nb_nb]:
                                if nb_nb_nb != v:
                                    self._nb_list[v].append(nb_nb_nb)
                                    for nb_nb_nb_nb in self._edgelist[nb_nb_nb]:
                                        if nb_nb_nb_nb != v:
                                            self._nb_list[v].append(nb_nb_nb_nb)
                else:
                    if len(self._edgelist[v]) == 1:
                        self._nb_list[v].append(v)

    def _get_nb_strategy5(self):
        self._nb_list = []

        for v in range(self._num_of_nodes):
            self._nb_list.append([])

            for nb in self._edgelist[v]:
                if nb != v:
                    self._nb_list[v].append(nb)
                    for nb_nb in self._edgelist[nb]:
                        if nb_nb != v:
                            self._nb_list[v].append(nb_nb)
                            for nb_nb_nb in self._edgelist[nb_nb]:
                                if nb_nb_nb != v:
                                    self._nb_list[v].append(nb_nb_nb)
                                    for nb_nb_nb_nb in self._edgelist[nb_nb_nb]:
                                        if nb_nb_nb_nb != v:
                                            self._nb_list[v].append(nb_nb_nb_nb)
                                            for nb_nb_nb_nb_nb in self._edgelist[nb_nb_nb_nb]:
                                                if nb_nb_nb_nb_nb != v:
                                                    self._nb_list[v].append(nb_nb_nb_nb_nb)
                else:
                    if len(self._edgelist[v]) == 1:
                        self._nb_list[v].append(v)

    def _get_nb_strategy2_itself(self):
        self._nb_list = []

        for v in range(self._num_of_nodes):
            self._nb_list.append([])
            for nb in self._edgelist[v]:
                self._nb_list[v].append(nb)
                for nb_nb in self._edgelist[nb]:
                    self._nb_list[v].append(nb_nb)

        #self._nb_list = np.unique(self._nb_list)


    def compute_table(self):
        pass


    def sigmoid(self, z):

        if z > 6:
            return 1.0
        elif z < -6:
            return 0.0
        else:
            return 1 / (1 + math.exp(-z))
    """
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

    def compute_gradient_ns(self, v, ns=5):
        global uni

        if len(self._nb_list[v]) == 0:
            return 0.0

        grad = 0.0

        #neg_choices = [r for r in range(self._num_of_nodes) if r not in self._nb_list]
        #neg_samples = np.random.choice(neg_choices, size=len(self._nb_list[v])*ns, replace=True)
        neg_samples = uni.sample(count=len(self._nb_list[v])*ns)

        
        #for n in neg_samples:
        #    if n in self._nb_list[v]:
        #        self.neg_count += 1
        
        #target = np.hstack((self._nb_list[v], neg_samples))
        #np.random.shuffle(target)

        for u, label in zip(self._nb_list[v] + neg_samples, [1.0]*len(self._nb_list[v]) + [0.0]*len(neg_samples)):
            z = np.dot(self._F[v, :], self._F[u, :])
            grad += ( label - self.sigmoid(z) ) * self._F[u, :]

        return grad
    """
    def save_f(self, filename):

        with open(filename, 'w') as f:
            f.write("{} {}\n".format(self._num_of_nodes, self._dim_size))
            for v in range(self._num_of_nodes):
                f.write("{} {}\n".format(str(v), " ".join([str(value) for value in self._F0[v, :]])))

    def run(self, starting_alpha, num_of_iterations):
        global uni

        #self._F = np.random.rand(self._num_of_nodes, self._dim_size)
        self._F0 = np.random.uniform(low=-0.5 / self._dim_size, high=0.5 / self._dim_size,
                                    size=(self._num_of_nodes, self._dim_size))

        self._F1 = np.zeros(shape=(self._num_of_nodes, self._dim_size), dtype=np.float)
        #self._current_f_sum = np.sum(self._F, 0)

        self._get_nb_strategy2()

        vocab = Vocab(self._num_of_nodes, freq=[float(len(self._edgelist[i])) for i in range(self._num_of_nodes)])
        uni = UnigramTable(vocab=vocab)



        alpha = starting_alpha

        for iter in range(num_of_iterations):
            print("Iter: {}".format(iter))

            if ( iter+1 ) % 100 == 0:
                self.save_f(filename="./outputs/citeseer_st3_" + str(iter+1) + ".embedding")
            """
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
            """
            perm = np.random.permutation(self._num_of_nodes)
            for v in perm:

                #if len(self._nb_list[v]) == 0:
                #    return 0.0

                neu1e = np.zeros(self._dim_size)

                # neg_choices = [r for r in range(self._num_of_nodes) if r not in self._nb_list]
                # neg_samples = np.random.choice(neg_choices, size=len(self._nb_list[v])*ns, replace=True)
                ns = 5
                neg_samples = uni.sample(count=len(self._nb_list[v]) * ns)

                for u, label in zip(self._nb_list[v] + neg_samples,
                                    [1.0] * len(self._nb_list[v]) + [0.0] * len(neg_samples)):
                    z = np.dot(self._F0[v, :], self._F1[u, :])
                    p = self.sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * self._F1[u, :]
                    self._F1[u, :] += g * self._F0[v, :]

                self._F0[v, :] += neu1e

        #epsilon = (2.0 * self._graph.number_of_edges()) / ( self._num_of_nodes * (self._num_of_nodes-1) )

        print("Total neg: {}".format(self.neg_count))

        return self._F0

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

start_time = time.time()
bg = myBigclam(nxg=g, dim_size=128)
F = bg.run(starting_alpha=0.001, num_of_iterations=10000)
print("Running time: {}".format(round(time.time() - start_time, 4)))

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


