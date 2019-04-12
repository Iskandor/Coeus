//
// Created by mpechac on 21. 3. 2017.
//

#include "CartPole.h"
#include "RandomGenerator.h"
#include "rk4.hpp"
#include <iostream>

float *derivs(float t, int n, float sensors[], float params[]) {
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

    float *result = new float[n];
    float F = params[0] * 10;
    float mp = params[1];
    float mc = params[2];
    float l = params[3];
    float g = params[4];

    float s = sensors[0];
    float ds = sensors[1];
    float theta = sensors[2];
    float dtheta = sensors[3];

    float sin_theta = (float)sin(theta);
    float cos_theta = (float)cos(theta);

    float u = dtheta;
    float du = (g * sin_theta * (mc + mp) - (F + mp * l * pow(dtheta, 2) * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * pow(cos_theta, 2));
    float v = ds;
    float dv = (F - mp * l * (du * cos_theta - (pow(dtheta, 2) * sin_theta))) / (mc + mp);

    result[0] = v;
    result[1] = dv;
    result[2] = u;
    result[3] = du;

    return result;
}

CartPole::CartPole(bool p_randomInit, float p_poleLength) : Environment() {
    _randomInit = p_randomInit;
    _indim = 1;
    _outdim = 4;
    L = p_poleLength;

    _params[0] = _action;
    _params[1] = MP;
    _params[2] = MC;
    _params[3] = L;
    _params[4] = G;
    _sensors = vector<float>(4);
}

CartPole::~CartPole() {

}

vector<float> CartPole::getSensors() {
    return vector<float>(_sensors);
}

void CartPole::performAction(float p_action) {
    _action = p_action;
    step();
}

void CartPole::reset() {
    if (_randomInit) {
        _angle = RandomGenerator::get_instance().random(-0.2f, 0.2f);
        _pos = RandomGenerator::get_instance().random(-0.5f, 0.5f);
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
    //float* new_state = RK4::rk4vec(0, _outdim, _sensors.data(), _params, DT, derivs);
    //_sensors = vector<float>(new_state, new_state + _outdim);
    step2();
    _pos = _sensors[0];
    _angle = _sensors[2];
    //delete[] new_state;
}

void CartPole::step2() {
    float F = _action * 1;

    float s = _sensors[0];
    float ds = _sensors[1];
    float theta = _sensors[2];
    float dtheta = _sensors[3];

    float sin_theta = (float)sin(theta);
    float cos_theta = (float)cos(theta);

    float temp = (F + MP * L * dtheta * dtheta * sin_theta) / (MP + MC);
    float thetaAcc = (G * sin_theta - cos_theta * temp) / (L * (4.0 / 3.0 - MP * cos_theta * cos_theta / (MP + MC)));
    float sAcc = temp - MP * L * thetaAcc * cos_theta / (MP + MC);

    s += DT * ds;
    ds += DT * sAcc;
    theta += DT * dtheta;
    dtheta += DT * thetaAcc;

    _sensors[0] = s;
    _sensors[1] = ds;
    _sensors[2] = theta;
    _sensors[3] = dtheta;
}
