//
// Created by mpechac on 21. 3. 2017.
//

#include "CartPole.h"
#include "RandomGenerator.h"

CartPole::CartPole(): _x(0), _x_dot(0), _theta(0), _theta_dot(0), _episode_length(0), _failed(false)
{
}

CartPole::~CartPole() = default;

vector<float> CartPole::get_state(bool p_norm) const
{
	vector<float> result(STATE);

	result[0] = p_norm ? _x / _x_threshold : _x;
	result[1] = p_norm ? _x_dot / 2 : _x_dot;
	result[2] = p_norm ? _theta / _theta_threshold_radians : _theta;
	result[3] = p_norm ? _theta_dot / 3 : _theta_dot;

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

	_episode_length++;
	_x = _x + _tau * _x_dot;
	_x_dot = _x_dot + _tau * _xacc;
	_theta = _theta + _tau * _theta_dot;
	_theta_dot = _theta_dot + _tau * thetaacc;
}


void CartPole::reset() {
	_failed = false;
	_episode_length = 0;
	_x = RandomGenerator::get_instance().random(-0.05f, 0.05f);
	_x_dot = RandomGenerator::get_instance().random(-0.05f, 0.05f);
	_theta = RandomGenerator::get_instance().random(-0.05f, 0.05f);
	_theta_dot = RandomGenerator::get_instance().random(-0.05f, 0.05f);
}

string CartPole::to_string() const
{	
    string s;
    s += std::to_string(_x);
    s += ' ';
    s += std::to_string(_x_dot);
    s += ' ';
    s += std::to_string(_theta);
    s += ' ';
    s += std::to_string(_theta_dot);
    s += '\n';

    return s;
}

bool CartPole::is_finished()
{
	bool done = false;
	
	done |= _x < -_x_threshold;
	done |= _x > _x_threshold;
	done |= _theta < -_theta_threshold_radians;
	done |= _theta > _theta_threshold_radians;

	if (done) _failed = true;
	
	done |= _episode_length > 200;

	return done;
}

float CartPole::get_reward() const
{
	float reward = 1;

	if (_failed) reward = -1.0;

	return reward;
}
