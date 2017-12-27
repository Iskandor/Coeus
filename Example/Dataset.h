//
// Created by user on 5. 11. 2017.
//

#ifndef NEURONET_DATASET_H
#define NEURONET_DATASET_H

#include <map>
#include "Sequence.h"

using namespace std;

namespace MNS {

class Dataset {

public:
    Dataset();
    ~Dataset();

    void loadData(string p_filename_v, string p_filename_m);
    vector<Sequence*>* permute();

private:
    map<int, map<int, Sequence*>> _buffer;
    vector<Sequence*> _permBuffer;

    void parseLines(vector<string> p_vLines, vector<string> p_mLines);

};

}



#endif //NEURONET_DATASET_H
