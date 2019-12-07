#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AMSGradRule : public IUpdateRule
	{
	public:
		AMSGradRule(ParamModel* p_model, float p_alpha, float p_beta1 = 0.9, float p_beta2 = 0.999, float p_epsilon = 1e-8);
		~AMSGradRule();

		void calc_update(Gradient& p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(ParamModel* p_model) override;

	private:
		void update_momentum(const string& p_id, Tensor &p_gradient);

		float _beta1;
		float _beta2;
		float _epsilon;

		map<string, Tensor> _m;
		map<string, Tensor> _v;
		map<string, Tensor> _v_mean;
		
	};
}