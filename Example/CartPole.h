//
// Created by mpechac on 21. 3. 2017.
//

#ifndef NEURONET_CARTPOLE_H
#define NEURONET_CARTPOLE_H

#include <string>
#include <vector>
#include <Coeus.h>

using namespace std;

float* derivs(float t, int n, float sensors[], float params[]);

class CartPole {
public:
    CartPole();
    ~CartPole();

    vector<float> get_state(bool p_norm = false) const;
    void perform_action(float p_action);
    void reset();

    string to_string() const;
	bool is_finished();
	float get_reward() const;

	static const int STATE = 4;
	static const int ACTION = 1;

private:

	float _gravity = 9.8f;
	float _masscart = 1.0f;
	float _masspole = 0.1f;
	float _total_mass = (_masspole + _masscart);
	float _length = 0.5f;
	float _polemass_length = (_masspole * _length);
	float _force_mag = 10.0f;
	float _tau = 0.02f;

	float _theta_threshold_radians = 12 * Coeus::PI / 180;
	float _x_threshold = 2.4f;

	float _x;
	float _x_dot;
	float _theta;
	float _theta_dot;
	int	_episode_length;
	bool _failed;
};


#endif //NEURONET_CARTPOLE_H
