#pragma once
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) BackPropRule : public IUpdateRule
	{
	public:
		BackPropRule(ParamModel* p_model, float p_alpha, float p_momentum = 0, bool p_nesterov = false);
		~BackPropRule();

		void calc_update(Gradient& p_gradient, float p_alpha = 0) override;
		IUpdateRule* clone(ParamModel* p_model) override;

	private:	
		float	_momentum;
		bool	_nesterov;
	};
}

