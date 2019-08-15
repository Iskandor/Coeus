#include "ReinforceRule.h"

using namespace Coeus;

ReinforceRule::ReinforceRule(NetworkGradient* p_network_gradient, float p_alpha) : IUpdateRule(p_network_gradient, p_alpha)
{
}

ReinforceRule::~ReinforceRule()
{
}

void ReinforceRule::calc_update(map<string, Tensor>* p_gradient, float p_delta, float p_alpha)
{
	_delta = p_delta;
	calc_update(p_gradient, p_alpha);
}

void ReinforceRule::calc_update(map<string, Tensor>* p_gradient, float p_alpha)
{

}

IUpdateRule* ReinforceRule::clone(NetworkGradient* p_network_gradient)
{
	return nullptr;
}

void ReinforceRule::reset()
{
}

