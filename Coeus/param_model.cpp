#include "param_model.h"



param_model::param_model()
= default;


param_model::~param_model()
{
	for(auto param : _model)
	{
		delete param.second;
	}
}

void param_model::copy_params(param_model& p_source, const float p_ratio)
{
	for (auto param : _model)
	{
		if (p_ratio < 1.f)
		{
			param.second->params() = p_source._model[param.first]->params() * p_ratio + param.second->params() * ( 1 - p_ratio);
		}
		else
		{
			param.second->params() = p_source._model[param.first]->params();
		}
		
	}
}

param* param_model::add_param(const std::initializer_list<int> p_shape)
{
	param* result = new param(p_shape);

	_model[result->id()] = result;

	return result;
}

param* param_model::add_param(param* p_param)
{
	_model[p_param->id()] = p_param;

	return p_param;
}

void param_model::add_model(param_model& p_model)
{
	for (auto param : p_model._model)
	{
		_model[param.first] = param.second;
	}
}

std::map<std::string, tensor> param_model::zero_model()
{
	std::map<std::string, tensor> result;

	for (auto param : _model)
	{
		result[param.first] = tensor::zero_like(param.second->gradient());
	}

	return result;
}
