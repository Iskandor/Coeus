#include "ParamModelStorage.h"
#include <algorithm>
#include <cassert>

using namespace Coeus;

ParamModelStorage& ParamModelStorage::instance()
{
	static ParamModelStorage instance;
	return instance;
}

void ParamModelStorage::bind(string& p_parent, string& p_child)
{
	_keys[p_parent].push_back(p_child);
}

void ParamModelStorage::add(string& p_parent, ParamModel* p_model)
{
	bool bond = true;

	for (const auto& k : _keys)
	{
		if (p_parent == k.first) bond = false;
	}

	if (!bond)
	{
		for (auto p : p_model->_params)
		{
			_storage[p.first] = p.second;
		}
	}
}

void ParamModelStorage::release(ParamModel* p_model)
{
	const string model_id = find_model(p_model->_id);

	if (!model_id.empty())
	{	
		_keys[model_id].erase(std::remove(_keys[model_id].begin(), _keys[model_id].end(), p_model->_id), _keys[model_id].end());

		if (_keys[model_id].empty())
		{
			for (const auto& p : p_model->_params)
			{
				delete p.second;
				_storage.erase(p.first);
			}
		}
	}
}

ParamModelStorage::ParamModelStorage()
= default;


ParamModelStorage::~ParamModelStorage()
{
	if (!_storage.empty())
	{
		assert(0, "ParamModelStorage: Memory leak");
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
