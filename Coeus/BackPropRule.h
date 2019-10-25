#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) BackPropRule : public IUpdateRule
	{
	public:
		BackPropRule(NetworkGradient* p_network_gradient, float p_alpha, float p_momentum = 0, bool p_nesterov = false);
		~BackPropRule();

		void calc_update(map<string, Tensor>& p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(NetworkGradient* p_network_gradient) override;

	private:	
		float	_momentum;
		bool	_nesterov;
	};
}

