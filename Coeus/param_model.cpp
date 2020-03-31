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

param* param_model::add_param(const std::initializer_list<int> p_shape)
{
	param* result = new param(p_shape);

	_model[result->id()] = result;

	return result;
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
