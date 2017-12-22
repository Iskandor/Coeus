//
// Created by mpechac on 10. 3. 2016.
//

#ifndef LIBNEURONET_DATASET_H
#define LIBNEURONET_DATASET_H

#include <vector>
#include "DatasetConfig.h"
#include <Tensor.h>

using namespace std;
using namespace FLAB;

class Dataset {
public:
    Dataset();
    ~Dataset();

    void load(string p_filename, DatasetConfig p_format);
    void normalize();

    vector<pair<Tensor*, Tensor*>>* getData() { return &_buffer; };
    void permute();
protected:
    virtual void parseLine(string p_line, string p_delim);
private:
    DatasetConfig _config;
    vector<pair<Tensor*, Tensor*>> _buffer;
};
#endif //LIBNEURONET_DATASET_H
