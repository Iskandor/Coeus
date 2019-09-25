#include "SimpleContinuousEnv.h"



SimpleContinuousEnv::SimpleContinuousEnv()
{
	_winning_position = 0;
	_position = 0.f;
	_target = 7.f;
}


SimpleContinuousEnv::~SimpleContinuousEnv()
= default;

float SimpleContinuousEnv::get_state() const
{
	return _position;
}

void SimpleContinuousEnv::perform_action(const float p_action)
{
	_position += p_action * 0.1;

	if (_position < 0) _position = 0;
	if (_position > 10) _position = 10;

	if (_position < _target + THETA && _position > _target - THETA) {
		_winning_position++;
	}
	else
	{
		_winning_position = 0;
	}
}

float SimpleContinuousEnv::get_reward() const
{
	float reward = 0;

	if (is_winner()) reward = 1.f;
	if (is_failed()) reward = -1.f;

	return reward;
}

bool SimpleContinuousEnv::is_finished() const
{
	return is_winner() || is_failed();
}

bool SimpleContinuousEnv::is_winner() const
{
	return _winning_position == 1;
}

bool SimpleContinuousEnv::is_failed() const
{
	return _position == 0.f || _position == 10.f;
}

void SimpleContinuousEnv::reset()
{
	_winning_position = 0;
	_position = 3.f;
}
