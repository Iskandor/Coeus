#include "param_model.h"



/**
 * \brief Default contructor
 */
param_model::param_model()
= default;


param_model::~param_model()
{
	for(auto param : _model)
	{
		delete param.second;
	}
}

/**
 * \brief Performs copying of parameters from source model
 * \param p_source source model of parameters
 * \param p_ratio ration of copy operator: new_params = source_params * p_ratio + old_params * (1 - p_ratio)
 */
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

/**
 * \brief Creates a new parameter in the model
 * \param p_shape shape of the new parameter
 * \return pointer to the new parameter instance
 */
param* param_model::add_param(const std::initializer_list<int> p_shape)
{
	param* result = new param(p_shape);

	_model[result->id()] = result;

	return result;
}

/**
 * \brief Add existing parameter to the model
 * \param p_param existing parameter instance
 * \return pointer to the added parameter (the same as in the argument)
 */
param* param_model::add_param(param* p_param)
{
	_model[p_param->id()] = p_param;

	return p_param;
}

/**
 * \brief Merge the parameters of existing model into the model
 * \param p_model source model
 */
void param_model::add_model(param_model& p_model)
{
	for (auto param : p_model._model)
	{
		_model[param.first] = param.second;
	}
}

/**
 * \brief Create the same parameters as has the model with zero values
 * \return map of the new zero model parameters
 */
std::map<std::string, tensor> param_model::zero_model()
{
	std::map<std::string, tensor> result;

	for (auto param : _model)
	{
		result[param.first] = tensor::zero_like(param.second->gradient());
	}

	return result;
}
