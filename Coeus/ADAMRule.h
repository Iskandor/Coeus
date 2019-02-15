#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) ADAMRule : public IUpdateRule
	{
	public:
		ADAMRule(NetworkGradient* p_network_gradient, float p_alpha, float p_beta1 = 0.9, float p_beta2 = 0.999, float p_epsilon = 1e-8);
		~ADAMRule();

		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void calc_update(map<string, Tensor>* p_gradient, float p_alpha = 0) override;
		void reset() override;

		void set_step(int p_t);

	private:
		int _t;
		float _beta1;
		float _beta2;
		float _epsilon;

		map<string, Tensor> _m;
		map<string, Tensor> _v;
		map<string, Tensor> _m_mean;
		map<string, Tensor> _v_mean;
	};
}

