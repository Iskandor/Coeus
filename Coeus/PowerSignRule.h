#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) PowerSignRule : public IUpdateRule
	{
	public:
		PowerSignRule(ParamModel* p_model, float p_alpha);
		~PowerSignRule();

		void calc_update(map<string, Tensor>& p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(ParamModel* p_model) override;

	private:
		map<string, Tensor> _m;
	};
}
