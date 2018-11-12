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
    vector <vector <int> > adjlist;

    void vector2Adjlist(vector <vector <int>>) {
        for(int i=0; i<num_of_nodes; i++) {
            adjlist.push_back(vector<int>);
        }

        for(int j=0; j<num_of_edges; j++) {
            adjlist[edges[j][0]].push_back(edges[j][1]);
        }


    }

public:
    void readEdgeList(string file_path, bool directed) {

        fstream fs(file_path, fstream::in);
        if(fs.is_open()) {
            int u, v;
            int maxNodeId=0, minNodeId = 0;


            while(fs >> u >> v) {
                vector <int> edge{u, v};
                edges.push_back(edge);
                num_of_edges++;

                if (u > maxNodeId) { maxNodeId = u; }
                if (v > maxNodeId) { maxNodeId = v; }
            }
            fs.close();

            num_of_nodes = (unsigned int ) maxNodeId - minNodeId; // Fix the minNodeId
            // Convert edges into adjacent list
            vector2Adjlist(edges);

        } else {
            cout << "An error occurred during reading file!" << endl;
        }

    }

    unsigned int getNumOfNodes() {

        return num_of_nodes;
    }

    unsigned int getNumOfEdges() {

        return num_of_nodes;
    }

    void getEdges() {

        for(int i=0; i<edges.size(); i++) {
            cout << edges[i][0] << " " << edges[i][1] << endl;
        }

    }
    void  printAdjList() {

        for(int i=0; i<num_of_nodes; i++) {
            cout << i << ": ";
            for(int j=0; j<adjlist[i].size(); j++) {
                cout << j << " ";
            }
            cout << endl;
        }

    }

};



int main() {

    Graph g;
    g.readEdgeList("/home/abdulkadir/Desktop/fast_bern/citeseer.edgelist", 0);
    int p = g.getNumOfNodes();
    cout << "Nodes: " << p << endl;

    g.getEdges();
    std::cout << "Hello, World!" << std::endl;

    g.printAdjList();

    return 0;
}