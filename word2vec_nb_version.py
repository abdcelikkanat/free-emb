import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array
import networkx as nx
from scipy.sparse import *

fi = None
fo= None
cbow= None
neg = None
dim =None
starting_alpha=None
win=None
min_count=None
num_processes=None
binary=None
vocab=None
syn0=None
syn1=None
table=None
global_word_count=None
vocab_size=None
table= None
num_of_iters = None
nxg = None

def get_nb_list(nxg):

    nb_list = {node: [] for node in nxg.nodes()}

    for node in nxg.nodes():
        for nb in nx.neighbors(nxg, node):
            if nb != node:
                nb_list[node].append(nb)
                for nb_nb in nx.neighbors(nxg, nb):
                    if nb_nb != node:
                        nb_list[node].append(nb_nb)

    return nb_list






class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))

                # assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1

                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items  # List of VocabItem objects
        self.vocab_hash = vocab_hash  # Mapping from each token to its index in vocab
        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        # assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print('Total words in training file: %d' % self.word_count)
        print('Total bytes in training file: %d' % self.bytes)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print('Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab, power = 0.75):
        self.vocab_size = len(vocab)

        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant

        # table_size = 1e8 # Length of the unigram table
        table_size = np.uint32(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def log_sigmoid(z):

    return -math.log(1.0 + math.exp(-z))


def ben_initialize():
    global dim, vocab_size, syn0, syn1, table, vocab

    print('Initializing unigram table')
    #table = UnigramTable(vocab)
    #vocab_size = table.vocab_size

    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim

    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)


def ben_train_process(pid):
    global table, num_of_iters
    # Set fi to point to the right chunk of training file
    #start = vocab.bytes / num_processes * pid
    #end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    #fi.seek(start)
    # print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

    global syn0, syn1





    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0)
        syn1 = np.ctypeslib.as_array(syn1)


    alpha = starting_alpha

    #word_count = 0
    #last_word_count = 0

    nodes = list(nxg.nodes())

    nb_list = get_nb_list(nxg)

    for iter in range(num_of_iters):
        for node in nodes:



            target_list = []
            label_list = []

            neg_list = [v for v in nxg.nodes() if v not in nb_list[node]]
            neg_list = np.random.choice(neg_list, size=neg)[0]

            for neg_sample in neg_list:
                target_list.append(neg_sample)
                label_list.append(0)

            for pos_sample in nb_list[node]:
                target_list.append(pos_sample)
                label_list.append(1)

            #perm = np.random.permutation(len(target_list))

            #target_list = np.asarray(target_list)[perm]
            #label_list = np.asarray(label_list)[perm]

            neu1e = np.zeros(dim)

            context_word = int(node)
            for target, label in zip(target_list, label_list):
                z = np.dot(syn0[context_word], syn1[int(target)])
                p = sigmoid(z)
                g = alpha * (label - p)
                neu1e += g * syn1[int(target)]  # Error to backpropagate to syn0
                syn1[int(target)] += g * syn0[context_word]  # Update syn1

            # Update syn0
            syn0[context_word] += neu1e


        # Recalculate alpha
        #alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
        print("Iter: {} alpha: {}".format(iter, alpha))
        #alpha = alpha / (1 + vocab_size * 1.0)
        alpha = 0.002
        #if alpha < starting_alpha * 0.001:  alpha = starting_alpha * 0.001

        if iter % 10 == 0:
            neg_log_error = 0.0
            for node in nodes:
                node_err = 0
                for nb in nb_list[node]:
                    z = np.dot(syn0[int(node)], syn1[int(nb)])
                    p = sigmoid(z)
                    node_err = np.log(p)

                    neg_log_error += - node_err

                neg_list = [v for v in nxg.nodes() if v not in nb_list[node]]
                neg_list = np.random.choice(neg_list, size=neg)[0]

                """
                for neg_node in neg_list:
                    z = np.dot(syn0[int(node)], syn1[int(neg_node)])
                    p = 1.0 - sigmoid(z)
                    if p < 0.000001:
                        p = 0.000001
                    node_err = np.log(np.exp(p))

                    neg_log_error += - node_err
                """
            print("Negative log error: {}".format(neg_log_error))


    #sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
    #                 (alpha, global_word_count.value, vocab.word_count,
    #                  float(global_word_count.value) / vocab.word_count * 100))
    sys.stdout.flush()
    #fi.close()


def ben_save(vocab, syn0, fo):
    print('Saving model to', fo)
    dim = len(syn0[0])

    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0), dim))
    for token, vector in zip(nxg.nodes(), syn0):
        word = str(token)
        if word not in ['<bol>', '<eol>', '<unk>']:
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()


def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi

    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)


def ben_train():
    global fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary, vocab_size, table, vocab

    global nxg

    global syn0, syn1

    # Read train file to init vocab
    #vocab = Vocab(fi, min_count)

    vocab_size = nxg.number_of_nodes()

    # Init net
    syn0, syn1 = ben_initialize()

    global_word_count = Value('i', 0)
    if neg > 0:
        pass
    else:
        print('Initializing Huffman tree')
        pass

    # Begin training using num_processes workers
    t0 = time.time()
    ben_train_process(0)
    #pool = Pool(processes=num_processes, initializer=__init_process,
    #            initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
    #                      win, num_processes, global_word_count, fi))
    #pool.map(ben_train_process, range(num_processes))
    t1 = time.time()

    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')

    # Save model to file
    ben_save(vocab, syn0, fo)


if __name__ == '__main__':

    argsfi = "./inputs/citeseer_node2vec.corpus"
    argsfo = "./outputs/citeseer_node2vec.embedding"

    nxg = nx.read_gml("./datasets/citeseer_undirected.gml")
    #nxg = nx.read_gml("./datasets/karate.gml")

    argscbow = 0
    argsneg = 5
    argsdim = 128
    argsalpha = 0.025
    argswin = 10
    argsmin_count = 0
    argsnum_processes = 1
    argsbinary = 0



    fi = argsfi
    fo = argsfo
    #cbow = None
    neg = 5
    dim = 128
    starting_alpha = 0.025
    alpha = starting_alpha
    win = 10
    min_count = 0
    num_processes = 1
    #binary = None
    #vocab = None
    #syn0 = None
    #syn1 = None
    #table = None
    #global_word_count = None
    #vocab_size = None

    #train(argsfi, argsfo, bool(argscbow), argsneg, argsdim, argsalpha, argswin,
    #      argsmin_count, argsnum_processes, bool(argsbinary))

    num_of_iters = 100

    ben_train()