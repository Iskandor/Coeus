#include "ParamModel.h"

using namespace Coeus;

ParamModel::ParamModel(): _size(0)
{
}

ParamModel::~ParamModel()
= default;

int ParamModel::get_params_size() const
{
	return _size;
}

Tensor* ParamModel::add_param(const string& p_id, Tensor* p_param)
{
	_params[p_id] = p_param;
	_size += p_param->size();

	return p_param;
}

void ParamModel::add_param(ParamModel* p_model)
{
	for (auto& param : p_model->_params)
	{
		add_param(param.first, param.second);
	}
}

map<string, Tensor> ParamModel::get_empty_params() const
{
	map<string, Tensor> result;

	for (const auto& param : _params)
	{
		result[param.first] = Tensor(param.second->rank(), param.second->shape(), Tensor::ZERO);
	}

	return result;
}
