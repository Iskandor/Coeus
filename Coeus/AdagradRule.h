#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AdagradRule : public IUpdateRule
	{
	public:
		AdagradRule(NetworkGradient* p_network_gradient, float p_alpha, float p_epsilon = 1e-8);
		~AdagradRule();

		void calc_update(map<string, Tensor>* p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;

	private:
		float _epsilon;
		map<string, Tensor> _G;
	};
}

