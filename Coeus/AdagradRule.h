#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) AdagradRule : public IUpdateRule
	{
	public:
		AdagradRule(NetworkGradient* p_network_gradient, double p_alpha, double p_epsilon);
		~AdagradRule();

		void calc_update() override;
	private:
		void init_structures() override;

		double _epsilon;
		map<string, Tensor> _G;
	};
}

