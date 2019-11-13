#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) RMSPropRule : public IUpdateRule
	{
	public:
		RMSPropRule(ParamModel* p_model, float p_alpha, float p_decay = 0.9, float p_epsilon = 1e-8);
		~RMSPropRule();

		void calc_update(Gradient& p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(ParamModel* p_model) override;

	private:
		float _decay;
		float _epsilon;

		map<string, Tensor> _cache;
	};
}
