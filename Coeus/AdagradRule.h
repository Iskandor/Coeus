#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AdagradRule : public IUpdateRule
	{
	public:
		AdagradRule(NetworkGradient* p_network_gradient, double p_alpha, double p_epsilon = 1e-8);
		~AdagradRule();

		void calc_update(map<string, Tensor>* p_gradient) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
	private:
		double _epsilon;
		map<string, Tensor> _G;
	};
}

