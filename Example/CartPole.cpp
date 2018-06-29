//
// Created by mpechac on 21. 3. 2017.
//

#include "CartPole.h"
#include "RandomGenerator.h"
#include "rk4.hpp"
#include <iostream>

using namespace FLAB;

double *derivs(double t, int n, double sensors[], double params[]) {
    /*
     * This function is needed for the Runge-Kutta integration approximation method. It calculates the
    derivatives of the state variables given in x. for each variable in x, it returns the first order
    derivative at time t.
    F = self.action
    (theta, theta_, _s, s_) = x
    u = theta_
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    mp = self.mp
    mc = self.mc
    l = self.l
    u_ = (self.g * sin_theta * (mc + mp) - (F + mp * l * theta_ ** 2 * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * cos_theta ** 2)
    v = s_
    v_ = (F - mp * l * (u_ * cos_theta - (theta_ ** 2 * sin_theta))) / (mc + mp)
    return (u, u_, v, v_)
     */

    double *result = new double[n];
    double F = params[0] * 10;
    double mp = params[1];
    double mc = params[2];
    double l = params[3];
    double g = params[4];

    double s = sensors[0];
    double ds = sensors[1];
    double theta = sensors[2];
    double dtheta = sensors[3];

    double sin_theta = (double)sin(theta);
    double cos_theta = (double)cos(theta);

    double u = dtheta;
    double du = (g * sin_theta * (mc + mp) - (F + mp * l * pow(dtheta, 2) * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * pow(cos_theta, 2));
    double v = ds;
    double dv = (F - mp * l * (du * cos_theta - (pow(dtheta, 2) * sin_theta))) / (mc + mp);

    result[0] = v;
    result[1] = dv;
    result[2] = u;
    result[3] = du;

    return result;
}

CartPole::CartPole(bool p_randomInit, double p_poleLength) : Environment() {
    _randomInit = p_randomInit;
    _indim = 1;
    _outdim = 4;
    L = p_poleLength;

    _params[0] = _action;
    _params[1] = MP;
    _params[2] = MC;
    _params[3] = L;
    _params[4] = G;
    _sensors = vector<double>(4);
}

CartPole::~CartPole() {

}

vector<double> CartPole::getSensors() {
    return vector<double>(_sensors);
}

void CartPole::performAction(double p_action) {
    _action = p_action;
    step();
}

void CartPole::reset() {
    if (_randomInit) {
        _angle = RandomGenerator::getInstance().random(-0.2, 0.2);
        _pos = RandomGenerator::getInstance().random(-0.5, 0.5);
    }
    else {
        _angle = -0.2;
        _pos = 0.2;
    }

    _t = 0;
    _sensors[0] = _pos;
    _sensors[1] = 0.0;
    _sensors[2] = _angle;
    _sensors[3] = 0.0;
}

string CartPole::toString() {
    string s;

    s += to_string(_sensors[0]);
    s += '\n';
    s += to_string(_sensors[1]);
    s += '\n';
    s += to_string(_sensors[2]);
    s += '\n';
    s += to_string(_sensors[3]);
    s += '\n';
    return s;
}

void CartPole::step() {
    _params[0] = _action;
    //cout << toString() << endl;
    //cout << _action << endl;
    //double* new_state = RK4::rk4vec(0, _outdim, _sensors.data(), _params, DT, derivs);
    //_sensors = vector<double>(new_state, new_state + _outdim);
    step2();
    _pos = _sensors[0];
    _angle = _sensors[2];
    //delete[] new_state;
}

void CartPole::step2() {
    double F = _action * 1;

    double s = _sensors[0];
    double ds = _sensors[1];
    double theta = _sensors[2];
    double dtheta = _sensors[3];

    double sin_theta = (double)sin(theta);
    double cos_theta = (double)cos(theta);

    double temp = (F + MP * L * dtheta * dtheta * sin_theta) / (MP + MC);
    double thetaAcc = (G * sin_theta - cos_theta * temp) / (L * (4.0 / 3.0 - MP * cos_theta * cos_theta / (MP + MC)));
    double sAcc = temp - MP * L * thetaAcc * cos_theta / (MP + MC);

    s += DT * ds;
    ds += DT * sAcc;
    theta += DT * dtheta;
    dtheta += DT * thetaAcc;

    _sensors[0] = s;
    _sensors[1] = ds;
    _sensors[2] = theta;
    _sensors[3] = dtheta;
}
