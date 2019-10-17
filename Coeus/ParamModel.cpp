#include "ParamModel.h"
#include "TensorOperator.h"

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

void ParamModel::DEBUG_compare(ParamModel* p_model)
{
	map<string, Tensor*> diff;

	for (auto& param : p_model->_params)
	{
		diff[param.first] = new Tensor(*param.second);
		TensorOperator::instance().vv_sub(_params[param.first]->arr(), param.second->arr(), diff[param.first]->arr(), param.second->size());
	}

	for(const auto& it : diff)
	{
		if (it.second->at(it.second->max_value_index()) != 0)
		{
			cout << it.first.c_str() << endl;
		}
	}
}

void ParamModel::polyak_averaging(const float p_polyak, ParamModel* p_model)
{
	for (auto it = p_model->_params.begin(); it != p_model->_params.end(); it++) {
	}
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
