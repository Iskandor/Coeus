#pragma once
#include "ienvironment.h"

class mountain_car : public ienvironment
{
public:
	mountain_car();
	~mountain_car() override;
	
	tensor get_state() override;
	void do_action(tensor& p_action) override;
	float get_reward() override;
	void reset() override;
	bool is_finished() override;
	void set_state(tensor& p_state);

private:
	bool	_done;
	float	_reward;
	float	_position;
	float	_velocity;
	int		_steps;

	const float POWER = 0.0015f;
	const float MIN_POSITION = -1.2f;
	const float MAX_POSITION = 0.6f;	
	const float VELOCITY_LIMIT = 0.07f;
};

