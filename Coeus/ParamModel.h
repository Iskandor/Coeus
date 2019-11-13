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
		vector<string>& ids() { return _ids; }

		friend ostream &operator<<(ostream &output, const ParamModel &p_model) {

			for (auto& _param : p_model._params)
			{
				cout << _param.first.c_str() << endl;
				cout << *_param.second << endl;
			}

			return output;
		}

		void DEBUG_compare(ParamModel* p_model);

		void polyak_averaging(float p_polyak, ParamModel* p_model);
		void copy_params(const ParamModel* p_model);
		void average_params(ParamModel** p_model, int p_size);

	protected:
		Tensor* add_param(const string& p_id, Tensor* p_param);
		void add_param(ParamModel* p_model);

		map<string, Tensor*> _params;
		vector<string>		 _ids;

	private:
		int _size{};
};
}

