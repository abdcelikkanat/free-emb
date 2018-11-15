
"""
def get_nb(edges):
    self._nb_list = []

    for v in range(self._num_of_nodes):
        self._nb_list.append([])

        for nb in self._edgelist[v]:
            if nb != v:
                self._nb_list[v].append(nb)
                for nb_nb in self._edgelist[nb]:
                    if nb_nb != v and len(self._edgelist[v]) < 5:
                        self._nb_list[v].append(nb_nb)
        self._nb_list[v].append(nb) # append one more time
        #self._nb_list[v].append(nb)

    perm = np.random.permutation(len(self._nb_list[v]))
    self._nb_list[v] = [self._nb_list[v][p] for p in perm]

"""

def say(int num, int nums[]):

    return nums[1]