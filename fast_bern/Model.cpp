//
// Created by abdulkadir on 13/11/18.
//

#include "Model.h"


Model::Model(Graph g, int dim) {
    dim_size = dim;
    num_of_nodes = g.getNumOfNodes();
    adj_list = g.getAdjList();

    emb0 = new double*[num_of_nodes];
    emb1 = new double*[num_of_nodes];
    for(int i=0; i<num_of_nodes; i++) {
        emb0[i] = new double[dim_size];
        emb1[i] = new double[dim_size];
    }
}

Model::~Model() {

    for(int i=0; i<num_of_nodes; i++) {
        delete [] emb0[i];
        delete [] emb1[i];
    }
    delete emb0;
    delete emb1;


}

void Model::readGraph(string file_path, string filetype, bool directed) {

    g.readGraph(file_path, filetype, directed);
    num_of_nodes = g.getNumOfNodes();
    adj_list = g.getAdjList();

}

void Model::getNeighbors() {

    nb_list.resize(num_of_nodes);

    int nb, nb_nb;

    for(int node=0; node<num_of_nodes; node++) {

        for(int nb_inx=0; nb_inx<adj_list[node].size(); nb_inx++) {

            nb = adj_list[node][nb_inx]; // Get 1-neighbor
            nb_list[node].push_back(nb); // Set nb

            for(int nb_nb_inx=0; nb_nb_inx<adj_list[nb].size(); nb_nb_inx++) {

                nb_nb = adj_list[nb][nb_nb_inx]; // Get 2-neighbor
                nb_list[node].push_back(nb_nb); // Set nb_nb
                // ############
                // Add one more time for each 2-neighbor
                nb_list[node].push_back(nb);
            }

        }

    }

}

double Model::sigmoid(double z) {

    if(z > 6)
        return 1.0;
    else if(z < -6)
        return 0.0;
    else
        return 1.0 / ( 1.0 +  exp(z));

}

void Model::initialize() {

    uniform_real_distribution<double> real_distr(-5.0 /dim_size , 5.0/dim_size);


    for(int i=0; i<num_of_nodes; i++) {
        for(int j=0; j<dim_size; j++) {
            emb0[i][j] = real_distr(generator);
            emb1[i][j] = 0.0;
        }
        cout << endl;

    }

}

