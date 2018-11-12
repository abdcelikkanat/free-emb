import networkx as nx
import os


class BaseGraph:
    def __init__(self):
        self.graph = None
        self.dataset_path = os.path.join(os.path.dirname(__file__), "../datasets")

    def retrieve_graph(self, graphname):

        if graphname == "blogcatalog":
            self.graph = nx.read_gml(os.path.join(self.dataset_path, "blogcatalog.gml"))


b = BaseGraph()
b.retrieve_graph("blogcatalog")

