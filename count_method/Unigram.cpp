//
// Created by abdulkadir on 13/11/18.
//

#include "Unigram.h"



Unigram::Unigram() {
    cout << "Unigram Constructor" << endl;
}



Unigram::Unigram(int size, vector<int> freq, double expon) {

    cout << "Filling unigram table!" << endl;

    double norm = 0.0;

    for(int j=0; j<size; j++) {
        norm += pow(freq[j], expon);
    }

    double p=0;
    int i=0;
    for(int j=0; j<size; j++) {
        p += pow((double)freq[j], expon) / norm;
        while(i < UNIGRAM_TABLE_SIZE && (double)i/UNIGRAM_TABLE_SIZE < p ) {
            table[i] = j;
            i++;
        }
    }

}

Unigram::~Unigram() {

}


void Unigram::sample(int count, int samples[]) {

    uniform_int_distribution<int> uni(0, UNIGRAM_TABLE_SIZE-1);


    for(int i=0; i<count; i++) {
        samples[i] = table[uni(generator)];
    }
}

