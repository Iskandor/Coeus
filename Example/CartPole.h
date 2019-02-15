//
// Created by mpechac on 21. 3. 2017.
//

#ifndef NEURONET_CARTPOLE_H
#define NEURONET_CARTPOLE_H

#include <string>
#include <math.h>
#include "Environment.h"

using namespace std;

float* derivs(float t, int n, float sensors[], float params[]);

class CartPole : public Environment {
public:
    CartPole(bool p_randomInit = false, float p_poleLength = 0.5);
    ~CartPole();

    vector<float> getSensors();
    void performAction(float p_action);
    void reset();

    string toString();

    inline float getPoleAngle() {
        return _angle;
    }

    inline float getCartPosition() {
        return _pos;
    }



private:
    void    step();
    void    step2();

private:
    float L;
    const float G = 9.81;
    const float MP = 0.1;
    const float MC = 1.0;
    const float DT = 0.02;

    bool _randomInit;

    float _params[5];
    float _action;
    float _angle;
    float _pos;
    float _t;
    vector<float> _sensors;
};


#endif //NEURONET_CARTPOLE_H
