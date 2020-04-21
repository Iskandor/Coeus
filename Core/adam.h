#pragma once
#include "optimizer.h"

class __declspec(dllexport) adam : public optimizer
{
public:
	adam(neural_network* p_model, float p_alpha, float p_weight_decay = 0.f, float p_beta1 = 0.9, float p_beta2 = 0.999, float p_epsilon = 1e-8);
	~adam();

	void update() override;

protected:
	std::map<std::string, tensor> _v;
	std::map<std::string, tensor> _m;

	int	_t;
	float _beta1;
	float _beta2;
	float _epsilon;
};

