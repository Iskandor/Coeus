#pragma once
#include "neural_network.h"

class __declspec(dllexport) optimizer
{
public:
	optimizer(neural_network* p_model, float p_alpha, float p_weight_decay = 0.f);
	virtual ~optimizer();

	virtual void update(tensor& p_loss);

protected:
	float			_alpha;
	float			_weight_decay;
	neural_network*	_model;
};

