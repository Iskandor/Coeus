#include "ParamModel.h"
#include "TensorOperator.h"
#include "IDGen.h"
#include "ParamModelStorage.h"

using namespace Coeus;

ParamModel::ParamModel(): _size(0)
{
	_id = IDGen::instance().next();
	_params = ParamModelStorage::instance().create(_id);
}

ParamModel::ParamModel(ParamModel& p_copy, const bool p_clone)
{
	_id = IDGen::instance().next();

	if (p_clone)
	{
		_size = 0;
		_params = ParamModelStorage::instance().create(_id);
		
		for(const auto& param : ParamModelStorage::instance().get(p_copy._id)->data)
		{
			add_param(param.first, new Tensor(*param.second));
		}
	}
	else
	{
		_params = ParamModelStorage::instance().bind(p_copy._id, _id);
		_size = p_copy._size;
	}	
}

ParamModel::~ParamModel()
{
	ParamModelStorage::instance().release(_id);
}

int ParamModel::get_params_size() const
{
	return _size;
}

void ParamModel::DEBUG_compare(ParamModel* p_model) const
{
	map<string, Tensor*> diff;

	for (auto& param : p_model->_params->data)
	{
		diff[param.first] = new Tensor(*param.second);
		TensorOperator::instance().vv_sub(_params->data[param.first]->arr(), param.second->arr(), diff[param.first]->arr(), param.second->size());
	}

	for(const auto& it : diff)
	{
		if (it.second->at(it.second->max_value_index()) != 0)
		{
			cout << it.first.c_str() << endl;
		}
	}
}

void ParamModel::polyak_averaging(const float p_polyak, ParamModel* p_model) const
{
	for (auto it = p_model->_params->data.begin(); it != p_model->_params->data.end(); ++it) {
		TensorOperator::instance().vv_add(_params->data[it->first]->arr(), p_polyak, it->second->arr(), (1 - p_polyak), _params->data[it->first]->arr(), _params->data[it->first]->size());
	}
}

void ParamModel::copy_params(const ParamModel* p_model) const
{
	for (auto it = p_model->_params->data.begin(); it != p_model->_params->data.end(); ++it) {
		_params->data[it->first]->override(it->second);
	}
}

void ParamModel::average_params(ParamModel** p_model, int p_size) const
{
	map<string, Tensor> result = get_empty_params();

	for(int i = 0; i < p_size; i++)
	{
		for (auto it = p_model[i]->_params->data.begin(); it != p_model[i]->_params->data.end(); ++it) {
			result[it->first] += *it->second;
		}
	}

	for (auto it = _params->data.begin(); it != _params->data.end(); ++it) {
		result[it->first] *= 1.f / p_size;
		it->second->override(&result[it->first]);
	}
}

Tensor* ParamModel::add_param(const string& p_id, Tensor* p_param)
{
	_params->data[p_id] = p_param;
	_size += p_param->size();

	return p_param;
}

void ParamModel::add_param(Param* p_param)
{
	add_param(p_param->get_id(), p_param->get_data());
}

void ParamModel::add_param(ParamModel* p_model)
{
	for (auto& param : p_model->_params->data)
	{
		add_param(param.first, param.second);
	}
}

void ParamModel::update(map<string, Tensor>* p_update) const
{
	for (const auto& param : _params->data)
	{
		TensorOperator::instance().vv_add(param.second->arr(), (*p_update)[param.first].arr(), param.second->arr(), param.second->size());
	}
}

map<string, Tensor> ParamModel::get_empty_params() const
{
	map<string, Tensor> result;

	for (const auto& param : _params->data)
	{
		result[param.first] = Tensor(param.second->rank(), param.second->shape(), Tensor::ZERO);
	}

	return result;
}
