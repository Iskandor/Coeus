#include "simple_continuous_env.h"
#include "metrics.h"
#include "random_generator.h"


simple_continuous_env::simple_continuous_env(): _steps(0)
{
	_state_dim = 1;
	_action_dim = 1;
	_position = 0.f;
	_target = 7.f;
}


simple_continuous_env::~simple_continuous_env()
= default;

tensor simple_continuous_env::get_state()
{
	return tensor({ _state_dim }, tensor::VALUE, _position);
}

void simple_continuous_env::do_action(tensor& p_action)
{
	float action = p_action[0];
	if (action > 1.f) action = 1.f;
	if (action < -1.f) action = -1.f;
	_position += action * 0.1f;

	if (_position < 0.f) _position = 0.f;
	if (_position > 10.f) _position = 10.f;
	
	_steps++;
}

float simple_continuous_env::get_reward()
{
	float reward = 0;

	if (is_failed())
	{
		reward = -1.f;
	}
	else
	{
		reward = metrics::gaussian_distance(_position, 0.4f, _target);
	}

	return reward;
}

float simple_continuous_env::get_reward(tensor& p_state)
{
	_position = p_state[0];
	return get_reward();
}

void simple_continuous_env::reset()
{
	_steps = 0;
	_position = random_generator::instance().random(2.f, 5.f);
}

bool simple_continuous_env::is_finished()
{
	return is_failed() || _steps == MAX_STEPS;
}

bool simple_continuous_env::is_failed() const
{
	return _position == 0.f || _position == 10.f;
}
