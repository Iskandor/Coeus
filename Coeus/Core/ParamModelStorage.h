#pragma once
#include <string>
#include <map>
#include "ParamModel.h"

using namespace std;

namespace Coeus
{
	class __declspec(dllexport) ParamModelStorage
	{
	public:
		static ParamModelStorage& instance();
		ParamsContainer* create(string& p_parent);
		ParamsContainer* bind(string& p_parent, string& p_child);
		ParamsContainer* get(string& p_parent);
		void add(string& p_parent, ParamsContainer* p_model);
		void release(string& p_parent);
	private:
		ParamModelStorage();
		~ParamModelStorage();

		string find_model(string& p_parent);

		map<string, vector<string>>			_keys;
		map<string, ParamsContainer*>	_storage;
	};
}
