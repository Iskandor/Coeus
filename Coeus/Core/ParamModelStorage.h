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
		void bind(string& p_parent, string& p_child);
		void add(string& p_parent, ParamModel* p_model);
		void release(ParamModel* p_model);
	private:
		ParamModelStorage();
		~ParamModelStorage();

		string find_model(string& p_parent);

		map<string, vector<string>>	_keys;
		map<string, Tensor*>		_storage;
	};
}
