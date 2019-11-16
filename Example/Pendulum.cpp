#include "Pendulum.h"
#include "RandomGenerator.h"

Pendulum::Pendulum():
	_theta(0),
	_thetadot(0),
	_action(0), 
	_step(0)
{
	_state_dim = 3;
	_action_dim = 1;

	_state = Tensor::Zero({_state_dim});
}

Pendulum::~Pendulum()
= default;

Tensor Pendulum::get_state()
{
	_state[0] = sin(_theta);
	_state[1] = cos(_theta);
	_state[2] = _thetadot;

	return _state;
}

void Pendulum::do_action(Tensor& p_action)
{
	_step++;
	_action = p_action[0] * _max_torque;

	_thetadot += (-3 * _g / (2 * _l)) * sin(_theta + Coeus::PI) + 3 / (_m * pow(_l, 2) * _action) * _dt;
	_theta += _thetadot * _dt;
	if (_thetadot > _max_torque) _thetadot = _max_torque;
	if (_thetadot < -_max_torque) _thetadot = -_max_torque;
}

float Pendulum::get_reward()
{
	return -(pow(angle_normalize(_theta), 2) + 0.1 * pow(_thetadot, 2) + 0.001 * pow(_action, 2));
}

void Pendulum::reset()
{
	_step = 0;
	_theta = RandomGenerator::get_instance().random(-Coeus::PI, Coeus::PI);
	_thetadot = RandomGenerator::get_instance().random(-1.f, 1.f);
}

bool Pendulum::is_finished()
{
	return _step > _max_steps - 1;
}

float Pendulum::angle_normalize(const float p_angle) const
{
	return (fmod((p_angle + Coeus::PI), (2 * Coeus::PI)) - Coeus::PI);
}
