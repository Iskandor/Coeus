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

		static IUpdateRule* create_rule(GRADIENT_RULE p_rule, NetworkGradient* p_network_gradient, float p_alpha);
	};
}


