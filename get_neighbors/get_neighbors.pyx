from __future__ import print_function
from libcpp cimport bool
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

def get_neighbors(nb_list, n, l):
    """
    Returns the nodes can be reached up to 'l' steps
    :param edges:
    :param n:
    :param l:
    :return:
    """

    if l = 1:
        pass
    else:
        


    cdef int *visited = <int *> PyMem_Malloc(n * sizeof(int))

    for i in range(n):
        visited[i] = 0

    print(visited[5])


    PyMem_Free(visited)



def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b



