#pragma once
#include "IEnvironment.h"

class SimpleContinuousEnv : public Coeus::IEnvironment
{
public:
	SimpleContinuousEnv();
	~SimpleContinuousEnv();

	Tensor get_state() override;
	void do_action(Tensor& p_action) override;
	float get_reward() override;
	float get_reward(Tensor& p_state);
	void reset() override;
	bool is_finished() override;
	
private:
	bool is_failed() const;
	
	float	_position;
	float	_target;
	int		_steps;
	
	const int MAX_STEPS = 100;
};

