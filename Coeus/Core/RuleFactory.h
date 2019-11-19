#pragma once
#include "Coeus.h"
#include "IUpdateRule.h"

namespace Coeus
{
	class __declspec(dllexport) RuleFactory
	{
	public:
		RuleFactory();
		~RuleFactory();

		static IUpdateRule* create_rule(GRADIENT_RULE p_rule, ParamModel* p_model, float p_alpha);
	};
}


