#include "cart_pole.h"

#include "random_generator.h"

cart_pole::cart_pole() : _x(0), _x_dot(0), _theta(0), _theta_dot(0), _done(false), _reward(0), _steps(0)
{
	_state_dim = 4;
	_action_dim = 1;
}

cart_pole::~cart_pole()
{
	
}

tensor cart_pole::get_state()
{
	tensor state({ _state_dim });

	state[0] = _x;
	state[1] = _x_dot;
	state[2] = _theta;
	state[3] = _theta_dot;
	
	return state;
}

void cart_pole::do_action(tensor& p_action)
{
	float force = p_action[0] * FORCE_MAG;
	float costheta = cos(_theta);
	float sintheta = sin(_theta);
	float temp = (force + POLEMASS_LENGTH * _theta_dot * _theta_dot * sintheta) / TOTAL_MASS;
	float thetaacc = (GRAVITY * sintheta - costheta* temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
	float xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

	_x = _x + TAU * _x_dot;
	_x_dot = _x_dot + TAU * xacc;
	_theta = _theta + TAU * _theta_dot;
	_theta_dot = _theta_dot + TAU * thetaacc;
	_steps++;

	const bool failed = _x < -X_THRESHOLD || _x > X_THRESHOLD || _theta < -THETA_THRESHOLD || _theta > THETA_THRESHOLD;
	_done = failed || _steps == STEP_LIMIT;
	

	_reward = 0.0f;
	
	if (failed)
	{
		_reward = -1.0f;
	}
}

float cart_pole::get_reward()
{
	return _reward;
}

void cart_pole::reset()
{
	_done = false;
	_steps = 0;
	_x = random_generator::instance().random(-0.05f, 0.05f);
	_x_dot = random_generator::instance().random(-0.05f, 0.05f);
	_theta = random_generator::instance().random(-0.05f, 0.05f);
	_theta_dot = random_generator::instance().random(-0.05f, 0.05f);	
}

bool cart_pole::is_finished()
{
	return _done;
}

void cart_pole::set_state(tensor& p_state)
{
	_x = p_state[0];
	_x_dot = p_state[1];
	_theta = p_state[2];
	_theta_dot = p_state[3];
}
