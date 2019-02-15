//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_ENVIRONMENT_H
#define NEURONET_ENVIRONMENT_H

#include <vector>

using namespace std;

class Environment {
public:
    Environment();
    virtual ~Environment();

    virtual vector<float> getSensors() = 0;
    virtual void performAction(float p_action) = 0;
    virtual void reset() = 0;

protected:
    int _indim;
    int _outdim;
    bool _discreteStates;
    bool _discreteActions;
    unsigned int _numActions;
};

#endif //NEURONET_ENVIRONMENT_H
