//
// Created by mpechac on 11. 3. 2016.
//

#ifndef LIBNEURONET_DATASETCONFIG_H
#define LIBNEURONET_DATASETCONFIG_H

#include <string>

using namespace std;

struct DatasetConfig {
    int     inDim;
    int     targetDim;
    string  delimiter;
    int     targetPos;
};

#endif //LIBNEURONET_DATASETCONFIG_H
