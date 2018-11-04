#include "ParamModel.h"

using namespace Coeus;

ParamModel::ParamModel(): 
	_size(0)
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
	for (auto& _param : p_model->_params)
	{
		_params[_param.first] = _param.second;
	}
	_size += p_model->_size;
}
