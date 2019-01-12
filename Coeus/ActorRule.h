#pragma once
#include "IUpdateRule.h"

namespace Coeus
{
	class __declspec(dllexport) ActorRule : public IUpdateRule
	{
	public:
		ActorRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, double p_alpha);
		~ActorRule();

		void calc_update(map<string, Tensor>* p_gradient, double p_delta, Tensor* p_policy);
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset() override;
	private:
		IUpdateRule* _rule;
	};
}
