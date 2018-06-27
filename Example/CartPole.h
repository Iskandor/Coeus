//
// Created by mpechac on 21. 3. 2017.
//

#ifndef NEURONET_CARTPOLE_H
#define NEURONET_CARTPOLE_H

#include <string>
#include <math.h>
#include "Environment.h"

using namespace std;

double* derivs(double t, int n, double sensors[], double params[]);

class CartPole : public Environment {
public:
    CartPole(bool p_randomInit = false, double p_poleLength = 0.5);
    ~CartPole();

    vector<double> getSensors();
    void performAction(double p_action);
    void reset();

    string toString();

    inline double getPoleAngle() {
        return _angle;
    }

    inline double getCartPosition() {
        return _pos;
    }



private:
    void    step();
    void    step2();

private:
    double L;
    const double G = 9.81;
    const double MP = 0.1;
    const double MC = 1.0;
    const double DT = 0.02;

    bool _randomInit;

    double _params[5];
    double _action;
    double _angle;
    double _pos;
    double _t;
    vector<double> _sensors;
};


#endif //NEURONET_CARTPOLE_H
