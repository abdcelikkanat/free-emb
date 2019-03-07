import networkx as nx
import numpy as np
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel







g = nx.Graph()
g.add_edges_from([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                  [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7],
                   [3,4]])

sentences = []
for node in g.nodes():
    sentences.append([str(nb) for nb in g.neighbors(node)])


# Create a corpus from a list of texts
common_dictionary = Dictionary(sentences)
common_corpus = [common_dictionary.doc2bow(text) for text in sentences]

lda = LdaModel(common_corpus, num_topics=2, eta=0.001, alpha=[0.001, 0.001])

s = lda.get_topic_terms(topicid=0, topn=g.number_of_nodes())

token2id = common_dictionary.token2id

id2node = {token2id[token]: token for token in token2id}


print(s)
print([(id2node[p[0]], p[1]) for p in s])