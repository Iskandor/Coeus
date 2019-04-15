#pragma once
#include <map>
#include <iostream>
#include "Tensor.h"
#include "Coeus.h"

using namespace std;

namespace Coeus {
class __declspec(dllexport) ParamModel
{
	public:
		ParamModel();
		virtual ~ParamModel();

		int get_params_size() const;
		map<string, Tensor> get_empty_params() const;

		friend ostream &operator<<(ostream &output, const ParamModel &p_model) {

			for (auto& _param : p_model._params)
			{
				cout << _param.first.c_str() << endl;
				cout << *_param.second << endl;
			}

			return output;
		}

	protected:
		Tensor* add_param(const string& p_id, Tensor* p_param);
		void add_param(ParamModel* p_model);

		map<string, Tensor*> _params;

	private:
		int _size;
};
}

