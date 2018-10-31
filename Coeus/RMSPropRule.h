#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) RMSPropRule : public IUpdateRule
	{
	public:
		RMSPropRule(NetworkGradient* p_network_gradient, double p_alpha, double p_decay, double p_epsilon);
		~RMSPropRule();

		void calc_update(map<string, Tensor>* p_gradient) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;

	private:
		double _decay;
		double _epsilon;

		map<string, Tensor> _cache;
	};
}
