#include <iostream>
#include "Graph.h"

using namespace std;




int main() {

    Graph g;
    g.readGraph("/home/abdulkadir/Desktop/free-emb/fast_bern/citeseer.edgelist", "edgelist", false);
    int p = g.getNumOfNodes();
    cout << "Nodes: " << p << endl;

    //g.getEdges();
    std::cout << "Hello, World!" << std::endl;

    //g.printAdjList();

    return 0;
}