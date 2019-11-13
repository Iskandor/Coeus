#include "RuleFactory.h"
#include "ADAMRule.h"
#include "AdadeltaRule.h"
#include "AdagradRule.h"
#include "AdaMaxRule.h"
#include "AMSGradRule.h"
#include "BackPropRule.h"
#include "NadamRule.h"
#include "RMSPropRule.h"

using namespace Coeus;

RuleFactory::RuleFactory()
= default;


RuleFactory::~RuleFactory()
= default;

IUpdateRule* RuleFactory::create_rule(GRADIENT_RULE p_rule, ParamModel* p_model, float p_alpha)
{
	IUpdateRule* result = nullptr;

	switch(p_rule)
	{
	case ADAM_RULE:
		result = new ADAMRule(p_model, p_alpha);
		break;
	case ADADELTA_RULE:
		result = new AdadeltaRule(p_model, p_alpha);
		break;
	case ADAGRAD_RULE:
		result = new AdagradRule(p_model, p_alpha);
		break;
	case ADAMAX_RULE:
		result = new AdaMaxRule(p_model, p_alpha);
		break;
	case AMSGRAD_RULE:
		result = new AMSGradRule(p_model, p_alpha);
		break;
	case BACKPROP_RULE:
		result = new BackPropRule(p_model, p_alpha);
		break;
	case NADAM_RULE:
		result = new NadamRule(p_model, p_alpha);
		break;
	case RMSPROP_RULE:
		result = new RMSPropRule(p_model, p_alpha);
		break;
	default: ;
	}

	return result;
}