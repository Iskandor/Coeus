#pragma once
#include <map>
#include "Tensor.h"
#include "ParamModel.h"

using namespace std;

namespace Coeus {

	class __declspec(dllexport) IUpdateRule
	{
	public:
		IUpdateRule(ParamModel* p_model, float p_alpha);
		virtual ~IUpdateRule();

		virtual void calc_update(map<string, Tensor>& p_gradient, float p_alpha = 0);
		virtual IUpdateRule* clone(ParamModel* p_model) = 0;

		map<string, Tensor>* get_update() { return &_update; }

	protected:
		float				_alpha;
		ParamModel*			_model;
		map<string, Tensor> _update;
		
	};
}

