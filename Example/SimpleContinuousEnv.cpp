#include "SimpleContinuousEnv.h"
#include "Metrics.h"


SimpleContinuousEnv::SimpleContinuousEnv(): _steps(0)
{
	_state_dim = 1;
	_action_dim = 1;
	_position = 0.f;
	_target = 7.f;
}


SimpleContinuousEnv::~SimpleContinuousEnv()
= default;

Tensor SimpleContinuousEnv::get_state()
{
	return Tensor({ _state_dim }, Tensor::VALUE, _position);
}

void SimpleContinuousEnv::do_action(Tensor& p_action)
{
	_position += p_action[0] * 0.1f;

	if (_position < 0) _position = 0;
	if (_position > 10) _position = 10;
	
	_steps++;
}

float SimpleContinuousEnv::get_reward()
{
	float reward = 0;

	if (is_failed())
	{
		reward = -1;
	}
	else
	{
		reward = Coeus::Metrics::gaussian_distance(_position, 1, _target);
	}

	return reward;
}

float SimpleContinuousEnv::get_reward(Tensor& p_state)
{
	_position = p_state[0];
	return get_reward();
}

void SimpleContinuousEnv::reset()
{
	_steps = 0;
	_position = 5.f;
}

bool SimpleContinuousEnv::is_finished()
{
	return is_failed() || _steps == MAX_STEPS;
}

bool SimpleContinuousEnv::is_failed() const
{
	return _position == 0.f || _position == 10.f;
}
