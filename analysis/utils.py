import networkx as nx


def getCommunities(g):

    node2comm = nx.get_node_attributes(g, 'community')

    num_of_coms = 0

    for node in g.nodes():
        comms = node2comm[node]
        if type(comms) is int:
            node2comm[node] = [comms]

        if max(node2comm[node])+1 > num_of_coms:
            num_of_coms = max(node2comm[node])+1

    return node2comm, num_of_coms
