#include "optimizer.h"


optimizer::optimizer(neural_network* p_model, const float p_alpha, const float p_weight_decay):
	_alpha(p_alpha), 
	_weight_decay(1 - p_weight_decay),
	_model(p_model)
{
}

optimizer::~optimizer()
= default;

void optimizer::update(tensor& p_loss)
{
	_model->backward(p_loss);
	if (_weight_decay < 1.f)
	{
		for (auto& param : *_model)
		{
			param.second->params() *= _weight_decay;
		}
	}	
}
