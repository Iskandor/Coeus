#pragma once
#include "optimizer.h"
#include "neural_network.h"

class COEUS_DLL_API sgd : public optimizer
{
public:
	sgd(neural_network* p_model, float p_alpha, float p_momentum = 0.f, bool p_nesterov = false, float p_weight_decay = 0.f);
	~sgd();

	void update() override;

private:

	std::map<std::string, tensor> _velocity;

	float	_momentum;
	bool	_nesterov;
};

