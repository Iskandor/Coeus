#pragma once
#include "IEnvironment.h"

class MountainCar : public Coeus::IEnvironment
{
public:
	MountainCar();
	~MountainCar() override;
	
	Tensor get_state() override;
	void do_action(Tensor& p_action) override;
	float get_reward() override;
	void reset() override;
	bool is_finished() override;

private:
	bool	_done;
	float	_reward;
	float	_position;
	float	_velocity;
	int		_steps;

	const float POWER = 0.0015f;
	const float MIN_POSITION = -1.2f;
	const float MAX_POSITION = 0.6f;
	
	const float POSITION_LIMIT = 1.2f;
	const float VELOCITY_LIMIT = 0.07f;
};

