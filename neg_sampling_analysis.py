import numpy as np
import networkx as nx
import math



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




vocab = Vocab(10, freq=[1.0 for i in range(10)])
for v in vocab:
    print(v)

uni = UnigramTable(vocab=vocab)
for x in range(100):
    print(uni.sample(count=10))