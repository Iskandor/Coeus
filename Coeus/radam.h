#pragma once
#include "adam.h"

class __declspec(dllexport) radam : public adam
{
public:
	radam(neural_network* p_model, float p_alpha, float p_weight_decay = 0.f, float p_beta1 = 0.9, float p_beta2 = 0.999, float p_epsilon = 1e-8);
	~radam();

	void update() override;
private:
	float _r;
	float _rho;
	float _rho_inf;
};

