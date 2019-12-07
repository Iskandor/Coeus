#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) RADAMRule : public IUpdateRule
	{
	public:
		RADAMRule(ParamModel* p_model, float p_alpha, float p_beta1 = 0.9, float p_beta2 = 0.999);
		~RADAMRule();

		IUpdateRule* clone(ParamModel* p_model) override;
		void calc_update(Gradient& p_gradient, float p_alpha = 0) override;

	private:
		int _t;
		float _beta1;
		float _denb1;
		float _beta2;
		float _denb2;
		float _rho;
		float _rho_inf;
		float _r;

		map<string, Tensor> _m;
		map<string, Tensor> _v;
		map<string, Tensor> _m_mean;
		map<string, Tensor> _v_mean;
	};
}
