#include "IUpdateRule.h"

using namespace Coeus;

IUpdateRule::IUpdateRule(NetworkGradient* p_network_gradient, const double p_alpha):
	_network_gradient(p_network_gradient),
	_learning_rate_module(nullptr),
	_alpha(p_alpha)
{
	_update = p_network_gradient->get_empty_params();
}

IUpdateRule::~IUpdateRule()
{
	delete _learning_rate_module;
}

void IUpdateRule::calc_update(map<string, Tensor>* p_gradient)
{
	if (_learning_rate_module != nullptr)
	{
		_alpha = _learning_rate_module->get_alpha();
	}
}


void IUpdateRule::init_learning_rate_module(ILearningRateModule* p_learning_rate_module)
{
	_learning_rate_module = p_learning_rate_module;
}
