#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) BackPropRule : public IUpdateRule
	{
	public:
		BackPropRule(NetworkGradient* p_network_gradient, double p_alpha, double p_momentum, bool p_nesterov);
		~BackPropRule();

		void calc_update(map<string, Tensor>* p_gradient) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;
	private:	
		double	_momentum;
		bool	_nesterov;
	};
}

