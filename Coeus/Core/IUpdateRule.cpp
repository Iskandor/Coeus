#include "IUpdateRule.h"

using namespace Coeus;

IUpdateRule::IUpdateRule(ParamModel* p_model, const float p_alpha):
	_alpha(p_alpha),
	_model(p_model)
{
	_update = p_model->get_empty_params();
}

IUpdateRule::~IUpdateRule()
= default;

void IUpdateRule::calc_update(Gradient& p_gradient, const float p_alpha)
{
	if (p_alpha > 0)
	{
		_alpha = p_alpha;
	}
}
