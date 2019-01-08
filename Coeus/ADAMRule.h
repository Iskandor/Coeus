#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) ADAMRule : public IUpdateRule
	{
	public:
		ADAMRule(NetworkGradient* p_network_gradient, double p_alpha, double p_beta1, double p_beta2, double p_epsilon);
		~ADAMRule();

		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void calc_update(map<string, Tensor>* p_gradient) override;
		void reset() override;

	private:
		void update_momentum(const string& p_id, Tensor &p_gradient);

		double _beta1;
		double _pow_beta1;
		double _beta2;
		double _pow_beta2;
		double _epsilon;

		map<string, Tensor> _m;
		map<string, Tensor> _v;
		map<string, Tensor> _m_mean;
		map<string, Tensor> _v_mean;
	};
}

