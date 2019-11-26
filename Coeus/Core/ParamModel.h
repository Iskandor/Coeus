#pragma once
#include <map>
#include <iostream>
#include "Tensor.h"
#include "Param.h"

using namespace std;

namespace Coeus {
class __declspec(dllexport) ParamModel
{
	friend class ParamModelStorage;
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

		void DEBUG_compare(ParamModel* p_model);

		void polyak_averaging(float p_polyak, ParamModel* p_model);
		
		void average_params(ParamModel** p_model, int p_size) const;
	
		Tensor* add_param(const string& p_id, Tensor* p_param);
		void	add_param(Param* p_param);
		void	add_param(ParamModel* p_model);
		void	update(map<string, Tensor> *p_update) const;

	protected:
		string				 _id;
		map<string, Tensor*> _params;

	private:
		int		_size;
};
}

