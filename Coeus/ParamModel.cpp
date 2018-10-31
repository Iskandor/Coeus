#include "ParamModel.h"

using namespace Coeus;

ParamModel::ParamModel()
= default;


ParamModel::~ParamModel()
= default;

Tensor* ParamModel::add_param(const string& p_id, Tensor* p_param)
{
	_params[p_id] = p_param;

	return p_param;
}

void ParamModel::add_param(ParamModel* p_model)
{
	for (auto& _param : p_model->_params)
	{
		_params[_param.first] = _param.second;
	}
}
