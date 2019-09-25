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

CartPole::CartPole() {
}

CartPole::~CartPole() = default;

vector<float> CartPole::get_state() const
{
	vector<float> result(STATE);

	result[0] = _x;
	result[1] = _x_dot;
	result[2] = _theta;
	result[3] = _theta_dot;

	return result;
}

void CartPole::perform_action(const float p_action)
{
	const float force = _force_mag * p_action;
	const float costheta = cos(_theta);
	const float sintheta = sin(_theta);
	const float temp = (force + _polemass_length * _theta_dot * _theta_dot * sintheta) / _total_mass;
	const float thetaacc = (_gravity * sintheta - costheta * temp) / (_length * (4.0 / 3.0 - _masspole * costheta * costheta / _total_mass));
	const float _xacc = temp - _polemass_length * thetaacc * costheta / _total_mass;

	_x = _x + _tau * _x_dot;
	_x_dot = _x_dot + _tau * _xacc;
	_theta = _theta + _tau * _theta_dot;
	_theta_dot = _theta_dot + _tau * thetaacc;
}


void CartPole::reset() {
	_x = RandomGenerator::get_instance().random(-0.05f, 0.05f);
	_x_dot = RandomGenerator::get_instance().random(-0.05f, 0.05f);
	_theta = RandomGenerator::get_instance().random(-0.05f, 0.05f);
	_theta_dot = RandomGenerator::get_instance().random(-0.05f, 0.05f);
}

string CartPole::to_string() const
{	
    string s;
    s += std::to_string(_x);
    s += '\n';
    s += std::to_string(_x_dot);
    s += '\n';
    s += std::to_string(_theta);
    s += '\n';
    s += std::to_string(_theta_dot);
    s += '\n';
    return s;
}

bool CartPole::is_finished() const
{
	bool done = false;
	
	done |= _x < -_x_threshold;
	done |= _x > _x_threshold;
	done |= _theta < -_theta_threshold_radians;
	done |= _theta > _theta_threshold_radians;

	return done;
}

float CartPole::get_reward()
{
	return 1.0;
}
