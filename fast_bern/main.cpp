#include <iostream>
#include "Graph.h"
#include "Unigram.h"
#include "Model.h"
#include <string>
#include <sstream>

using namespace std;




int main() {

    stringstream graph_path, embedding;
    string dataset = "blogcatalog";

    graph_path << "/home/abdulkadir/Desktop/free-emb/fast_bern/" << dataset << ".edgelist";
    //graph_path << "./" << dataset << ".edgelist";

    Graph g;
    g.readGraph(graph_path.str(), "edgelist", false);

    cout << "Number of nodes: " << g.getNumOfNodes() << endl;
    cout << "Number of edges: " << g.getNumOfEdges() << endl;


    Model model(g, 128);
    model.run(0.005, 1000, 5, 50, dataset);
    model.save_embeddings("./blogcatalog.embedding");


    return 0;
}