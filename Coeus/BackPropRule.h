#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) BackPropRule : public IUpdateRule
	{
	public:
		BackPropRule(NetworkGradient* p_network_gradient, double p_alpha, double p_momentum = 0, bool p_nesterov = false);
		~BackPropRule();

		void calc_update(map<string, Tensor>* p_gradient, double p_alpha = 0) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
		void reset() override;

	private:	
		double	_momentum;
		bool	_nesterov;
	};
}

