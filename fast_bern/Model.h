//
// Created by abdulkadir on 13/11/18.
//

#ifndef FAST_BERN_MODEL_H
#define FAST_BERN_MODEL_H

#include "Graph.h"
#include <vector>
#include "math.h"
#include <random>


using namespace std;

class Model {
    Graph g;
    unsigned int num_of_nodes = 0;
    unsigned int dim_size = 0;
    vector <vector <int> > adj_list;
    vector <vector <int>> nb_list;
    default_random_engine generator;


    double **emb0, **emb1;


public:
    Model(Graph g, int dim);
    ~Model();

    void initialize();
    double sigmoid(double z);
    void getNeighbors();
    void readGraph(string file_path, string filetype, bool directed);

};


#endif //FAST_BERN_MODEL_H
