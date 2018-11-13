#include <iostream>
#include "Graph.h"
#include "Unigram.h"
#include "Model.h"

using namespace std;




int main() {

    Graph g;
    g.readGraph("/home/abdulkadir/Desktop/free-emb/fast_bern/citeseer.edgelist", "edgelist", false);
    int p = g.getNumOfNodes();
    cout << "Nodes: " << p << endl;

    //g.getEdges();
    std::cout << "Hello, World!" << std::endl;

    //g.printAdjList();



    /*

    int size = 10;
    int count = 5;
    int samples[count] = {0};


    int *freq = new int[size];
    for(int i=0; i<size; i++) {
        freq[i] = 1;
    }


    Unigram uni(size, freq, 0.75);


    for(int j=0; j<100; j++) {
        uni.sample(count, samples);
        for (int i = 0; i < count; i++) {
            cout << samples[i] << " ";
        }
        cout << endl;

    delete []freq;
    }
    */
    Model my(g, 1);
    my.initialize();




    /* */
    return 0;
}