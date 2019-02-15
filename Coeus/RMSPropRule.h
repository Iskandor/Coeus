#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) RMSPropRule : public IUpdateRule
	{
	public:
		RMSPropRule(NetworkGradient* p_network_gradient, float p_alpha, float p_decay = 0.9, float p_epsilon = 1e-8);
		~RMSPropRule();

		void calc_update(map<string, Tensor>* p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset() override;

	private:
		float _decay;
		float _epsilon;

		map<string, Tensor> _cache;
	};
}
