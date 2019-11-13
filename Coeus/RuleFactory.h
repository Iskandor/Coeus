#pragma once
#include "Coeus.h"
#include "IUpdateRule.h"

namespace Coeus
{
	class RuleFactory
	{
	public:
		RuleFactory();
		~RuleFactory();

		static IUpdateRule* create_rule(GRADIENT_RULE p_rule, ParamModel* p_model, float p_alpha);
	};
}


