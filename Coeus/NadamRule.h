#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) NadamRule : public IUpdateRule
	{
	public:
		NadamRule(NetworkGradient* p_network_gradient, double p_alpha, double p_beta1, double p_beta2, double p_epsilon);
		~NadamRule();

		void calc_update(map<string, Tensor>* p_gradient) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
	private:
		void update_momentum(const string& p_id, Tensor &p_gradient);

		double _beta1;
		double _beta2;
		double _epsilon;

		map<string, Tensor> _m;
		map<string, Tensor> _v;
		map<string, Tensor> _m_mean;
		map<string, Tensor> _v_mean;
	};
}
