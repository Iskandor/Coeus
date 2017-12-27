//
// Created by user on 5. 11. 2017.
//

#ifndef NEURONET_SEQUENCE_H
#define NEURONET_SEQUENCE_H

#include <vector>
#include <map>
#include "Tensor.h"

using namespace std;
using namespace FLAB;

namespace MNS {

class Sequence {
public:
    Sequence(int p_id, int p_grasp);
    ~Sequence();

    void addMotorData(Tensor *p_data);
    void addVisualData(int p_perspective, Tensor *p_data);

    vector<Tensor*>* getMotorData();
    vector<Tensor*>* getVisualData();
    vector<Tensor*>* getVisualData(int p_perspective);

    int getGrasp() {return _grasp;};

private:
    int _id;
    int _grasp;

    map<int, vector<Tensor*>> _v_buffer;
    vector<Tensor*> _v_data;
    vector<Tensor*> _m_data;
};

}



#endif //NEURONET_SEQUENCE_H
