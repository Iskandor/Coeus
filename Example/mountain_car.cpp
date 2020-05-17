#include "mountain_car.h"
#include <algorithm>
#include "random_generator.h"


mountain_car::mountain_car(): _done(false), _reward(0), _position(0), _velocity(0), _steps(0)
{
	_state_dim = 2;
	_action_dim = 1;
}

mountain_car::~mountain_car()
= default;

tensor mountain_car::get_state()
{
	tensor state({ _state_dim });

	state[0] = _position;
	state[1] = _velocity;
	
	return state;
}

void mountain_car::do_action(tensor& p_action)
{
	_steps++;

	const float force = min<float>(max<float>(p_action[0], -1.0), 1.0);

	_velocity += force * POWER - 0.0025f * cos(3 * _position);
	
	if (_velocity > VELOCITY_LIMIT) _velocity = VELOCITY_LIMIT;
	if (_velocity < -VELOCITY_LIMIT) _velocity = -VELOCITY_LIMIT;
	_position += _velocity;
	
	if (_position > MAX_POSITION) _position = MAX_POSITION;
	if (_position < MIN_POSITION) _position = MIN_POSITION;
	if (_position == MIN_POSITION && _velocity < 0) _velocity = 0;

	_reward = -pow(p_action[0], 2) * 0.1f;

	_done = _position >= 0.45f && _velocity >= 0;

	if (_done)
	{
		_reward += 100;
	}
	_done |= _steps > 999;
}

float mountain_car::get_reward()
{
	return _reward;
}

void mountain_car::reset()
{
	_done = false;
	_reward = 0;
	_steps = 0;
	_position = random_generator::instance().random(-0.6f, -0.4f);
	_velocity = 0;
}

bool mountain_car::is_finished()
{
	return _done;
}

void mountain_car::set_state(tensor& p_state)
{
	_position = p_state[0];
	_velocity = p_state[1];
}
