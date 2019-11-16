#pragma once
#include "IEnvironment.h"

class Pendulum : Coeus::IEnvironment
{
public:
	Pendulum();
	~Pendulum();

	Tensor get_state() override;
	void do_action(Tensor& p_action) override;
	float get_reward() override;
	void reset() override;
	bool is_finished() override;

private:
	float angle_normalize(float p_angle) const;

	const int _max_steps = 25;
	const float _max_speed = 8;
	const float _max_torque = 2;
	const float _dt = .05;
	const float _g = 10;
	const float _m = 1.;
	const float _l = 1.;

	float _theta;
	float _thetadot;

	Tensor	_state;
	float	_action;
	int		_step;
};
