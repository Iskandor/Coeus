//
// Created by mpechac on 21. 3. 2017.
//

#ifndef NEURONET_CARTPOLE_H
#define NEURONET_CARTPOLE_H

#include <string>
#include "Coeus.h"
#include <vector>

using namespace std;

float* derivs(float t, int n, float sensors[], float params[]);

class CartPole {
public:
    CartPole();
    ~CartPole();

    vector<float> get_state() const;
    void perform_action(float p_action);
    void reset();

    string to_string() const;
	bool is_finished() const;
	float get_reward();

	static const int STATE = 4;
	static const int ACTION = 1;

private:

	float _gravity = 9.8;
	float _masscart = 1.0;
	float _masspole = 0.1;
	float _total_mass = (_masspole + _masscart);
	float _length = 0.5;
	float _polemass_length = (_masspole * _length);
	float _force_mag = 10.0;
	float _tau = 0.02;

	float _theta_threshold_radians = 12 * 2 * Coeus::PI / 360;
	float _x_threshold = 2.4;

	float _x;
	float _x_dot;
	float _theta;
	float _theta_dot;
};


#endif //NEURONET_CARTPOLE_H
