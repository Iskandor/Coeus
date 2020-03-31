#pragma once
#include "ienvironment.h"
#include "tensor.h"

class simple_continuous_env : public ienvironment
{
public:
	simple_continuous_env();
	~simple_continuous_env();

	tensor get_state() override;
	void do_action(tensor& p_action) override;
	float get_reward() override;
	float get_reward(tensor& p_state);
	void reset() override;
	bool is_finished() override;
	
private:
	bool is_failed() const;
	
	float	_position;
	float	_target;
	int		_steps;
	
	const int MAX_STEPS = 100;
};

