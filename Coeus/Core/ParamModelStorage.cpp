#include "ParamModelStorage.h"
#include <algorithm>

using namespace Coeus;

ParamModelStorage& ParamModelStorage::instance()
{
	static ParamModelStorage instance;
	return instance;
}

ParamsContainer* ParamModelStorage::create(string& p_parent)
{
	const auto model = new ParamsContainer();

	_storage[model->id] = model;
	_keys[model->id].push_back(p_parent);

	return model;
}

ParamsContainer* ParamModelStorage::bind(string& p_parent, string& p_child)
{
	const string model_id = find_model(p_parent);
	ParamsContainer* result = nullptr;

	if (!model_id.empty())
	{
		result = _storage[model_id];
		_keys[model_id].push_back(p_child);
	}

	return result;
}

ParamsContainer* ParamModelStorage::get(string& p_parent)
{
	ParamsContainer* result = nullptr;
	const string model_id = find_model(p_parent);

	result = _storage[model_id];

	return result;
}

void ParamModelStorage::add(string& p_parent, ParamsContainer* p_model)
{
	_storage[p_model->id] = p_model;
	_keys[p_model->id].push_back(p_parent);
}

void ParamModelStorage::release(string& p_parent)
{
	const string model_id = find_model(p_parent);

	if (!model_id.empty())
	{	
		_keys[model_id].erase(std::remove(_keys[model_id].begin(), _keys[model_id].end(), p_parent), _keys[model_id].end());

		if (_keys[model_id].empty())
		{
			delete _storage[model_id];
			_storage[model_id] = nullptr;
		}
	}
}

ParamModelStorage::ParamModelStorage()
= default;


ParamModelStorage::~ParamModelStorage()
{
	// this is security fallback, _storage should be full of nullptrs
	for(const auto& model : _storage)
	{
		delete model.second;
	}
}

string ParamModelStorage::find_model(string& p_parent)
{
	string model_id;

	for (const auto& model : _keys)
	{
		for (const auto& key : model.second)
		{
			if (key == p_parent)
			{
				model_id = model.first;
			}
		}
	}

	return model_id;
}
