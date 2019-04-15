#include "IUpdateRule.h"

using namespace Coeus;

IUpdateRule::IUpdateRule(NetworkGradient* p_network_gradient, const float p_alpha):
	_alpha(p_alpha),
	_network_gradient(p_network_gradient)
{
	_update = p_network_gradient->get_network()->get_empty_params();
}

IUpdateRule::~IUpdateRule()
= default;

void IUpdateRule::calc_update(map<string, Tensor>* p_gradient, const float p_alpha)
{
	if (p_alpha > 0)
	{
		_alpha = p_alpha;
	}
}
