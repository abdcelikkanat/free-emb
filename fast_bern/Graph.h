//
// Created by abdulkadir on 13/11/18.
//

#ifndef FAST_BERN_GRAPH_H
#define FAST_BERN_GRAPH_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class Graph {

private:
    unsigned int num_of_nodes = 0;
    unsigned int num_of_edges = 0;
    vector <vector <int> > edges;
    vector < vector <int> > adjlist;
    bool directed = 0;

    void vector2Adjlist(bool directed);

public:
    Graph();
    ~Graph();


    void readEdgeList(string file_path, bool directed);

    void readGraph(string file_path, string filetype, bool directed);

    unsigned int getNumOfNodes();

    unsigned int getNumOfEdges();

    void getEdges();

    void  printAdjList();

};



#endif //FAST_BERN_GRAPH_H
