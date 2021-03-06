//
// Created by abdulkadir on 13/11/18.
//

#include "Model.h"


Model::Model(Graph graph, int dim) {
    g = graph;
    dim_size = dim;
    num_of_nodes = graph.getNumOfNodes();
    adj_list = graph.getAdjList();
    getNeighbors();

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
                //nb_list[node].push_back(nb_nb); // Set nb_nb
                // ############
                // Add one more time for each 2-neighbor
                //nb_list[node].push_back(nb);
            }

        }

    }

}


void Model::getNeighbors_strategy1() {

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
        return 1.0 / ( 1.0 +  exp(-z));

}

void Model::run(double starting_alpha, int num_of_iters, int negative, int save_step, string save_file) {

    // Initialize parameters
    uniform_real_distribution<double> real_distr(-5.0 /dim_size , 5.0/dim_size);

    for(int node=0; node<num_of_nodes; node++) {
        for(int d=0; d<dim_size; d++) {
            emb0[node][d] = real_distr(generator);
            emb1[node][d] = 0.0;
        }
    }

    //save_embeddings("./citeseer0.embedding");

    // Set up sampling class
    vector <int> freq = g.getDegreeSequence();
    Unigram uni(num_of_nodes, freq, 0.75);

    stringstream embed_file_path;
    int *samples, *labels;
    int target, neg_sample_size, pos_sample_size, target_seq_size, label;
    double alpha, z, g, *neule;

    alpha = starting_alpha;

    for(int iter=0; iter<num_of_iters; iter++) {

        cout << "Iteration: " << iter << endl;

        for(int node=0; node<num_of_nodes; node++) {

            neg_sample_size = (int)nb_list[node].size()*negative;
            pos_sample_size = (int)nb_list[node].size();
            target_seq_size =  neg_sample_size + pos_sample_size;

            samples = new int[target_seq_size];
            labels = new int[target_seq_size];

            // set negative samples
            uni.sample(neg_sample_size, samples);
            //
            for(int l=0; l<neg_sample_size; l++)
                labels[l] = 0;
            //memset(labels, 0, (size_t)neg_sample_size);

            // set positive samples
            for(int j=neg_sample_size; j<target_seq_size; j++)
                samples[j] = nb_list[node][j-neg_sample_size];
            //
            for(int l=neg_sample_size; l<target_seq_size; l++)
                labels[l] = 1;
            //memset(labels+neg_sample_size, 1, (size_t)pos_sample_size);

            neule = new double[dim_size];
            //
            for(int d=0; d<dim_size; d++)
                neule[d] = 0.0;
            //memset(neule, 0, (size_t)dim_size);

            for(int j=0; j<target_seq_size; j++) {

                target = samples[j];
                label = labels[j];

                z = 0.0;
                for(int d=0; d<dim_size; d++)
                    z += emb0[node][d] * emb1[target][d];

                z = sigmoid(z);

                g = alpha * (label - z);

                for(int d=0; d<dim_size; d++) {
                    neule[d] += g * emb1[target][d];
                }

                for(int d=0; d<dim_size; d++)
                    emb1[target][d] += g*emb0[node][d];

            }

            for(int d=0; d<dim_size; d++)
                emb0[node][d] += neule[d];



            delete [] samples;
            delete [] labels;
            delete [] neule;
        }


        if((iter+1) % save_step == 0) {
            embed_file_path.str("");
            embed_file_path << save_file + "_" << iter+1 << ".embedding";
            save_embeddings(embed_file_path.str());
        }

    }

}


void Model::save_embeddings(string file_path) {

    fstream fs(file_path, fstream::out);
    if(fs.is_open()) {
        fs << num_of_nodes << " " << dim_size << endl;
        for(int node=0; node<num_of_nodes; node++) {
            fs << node << " ";
            for(int d=0; d<dim_size; d++) {
                fs << emb0[node][d] << " ";
            }
            fs << endl;
        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}