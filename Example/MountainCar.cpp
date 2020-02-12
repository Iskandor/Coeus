#include "MountainCar.h"
#include "RandomGenerator.h"
#include <algorithm>


MountainCar::MountainCar(): _position(0), _velocity(0)
{
	_state_dim = 2;
	_action_dim = 1;
}

MountainCar::~MountainCar()
= default;

Tensor MountainCar::get_state()
{
	Tensor state({ _state_dim }, Tensor::ZERO);

	state[0] = _position;
	state[1] = _velocity;
	
	return state;
}

void MountainCar::do_action(Tensor& p_action)
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

	_reward -= pow(p_action[0], 2) * 0.1f;

	_done = _position >= 0.45f && _velocity >= 0;

	if (_done)
	{
		_reward += 100;
	}
	_done |= _steps > 999;
}

float MountainCar::get_reward()
{
	return _reward;
}

void MountainCar::reset()
{
	_done = false;
	_reward = 0;
	_steps = 0;
	_position = RandomGenerator::get_instance().random(-0.6f, -0.4f);
	_velocity = 0;
}

bool MountainCar::is_finished()
{
	return _done;
}
