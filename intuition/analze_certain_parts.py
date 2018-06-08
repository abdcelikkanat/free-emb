import networkx as nx
import numpy as np

from base.base import *


class Analyze1(BaseGraph):

    def __init__(self):
        self.graph = self.retrieve_graph("blogcatalog")
        BaseGraph.__init__(self)

    def test_random(self):
        N = self.graph.number_of_nodes()

        chosen_node = np.random.choice(self.graph.nodes())
        print(chosen_node)


aa = Analyze1()
aa.test_random()