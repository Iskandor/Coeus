#pragma once
#include "IUpdateRule.h"

namespace Coeus
{
	class __declspec(dllexport) ReinforceRule : public IUpdateRule
	{
	public:
		ReinforceRule(NetworkGradient* p_network_gradient, float p_alpha);
		~ReinforceRule();

		void calc_update(map<string, Tensor>* p_gradient, float p_delta, float p_alpha);
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset() override;

	private:
		void calc_update(map<string, Tensor>* p_gradient, float p_alpha) override;

		float _delta;
	};
}


