#include "IUpdateRule.h"

using namespace Coeus;

IUpdateRule::IUpdateRule(NetworkGradient* p_network_gradient, const double p_alpha)
{
	_network_gradient = p_network_gradient;
	_update = p_network_gradient->get_empty_params();
	_alpha = p_alpha;
}


IUpdateRule::~IUpdateRule()
= default;

void IUpdateRule::calc_update()
{
}