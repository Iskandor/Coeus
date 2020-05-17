#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include "ienvironment.h"

class cart_pole : public ienvironment {
public:
	cart_pole();
	~cart_pole();

	tensor get_state() override;
	void do_action(tensor& p_action) override;
	float get_reward() override;
	void reset() override;
	bool is_finished() override;
	void set_state(tensor& p_state);
	
private:	
	float	_x;
	float	_x_dot;
	float	_theta;
	float	_theta_dot;
	bool	_done;
	float	_reward;
	int		_steps;
	
	const float GRAVITY = 9.8f;
	const float MASSCART = 1.0f;
	const float MASSPOLE = 0.1f;
	const float LENGTH = 0.5f;
	const float FORCE_MAG = 10.0f;
	const float TAU = 0.02f;
	const float TOTAL_MASS = MASSPOLE + MASSCART;
	const float POLEMASS_LENGTH = MASSPOLE * LENGTH;

	const float THETA_THRESHOLD = 12 * 2 * M_PI / 360;
	const float X_THRESHOLD = 2.4f;
	const int STEP_LIMIT = 200;
};
