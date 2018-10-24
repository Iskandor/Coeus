#include "IUpdateRule.h"

using namespace Coeus;

IUpdateRule::IUpdateRule(NetworkGradient* p_network_gradient, const double p_alpha)
{
	_network_gradient = p_network_gradient;
	_alpha = p_alpha;
	_init_structures = false;
}


IUpdateRule::~IUpdateRule()
= default;

void IUpdateRule::calc_update()
{
	if (!_init_structures) {
		init_structures();
	}

	for (auto it = _network_gradient->get_b_gradient()->begin(); it != _network_gradient->get_b_gradient()->end(); ++it) {
		_update[it->first] = -_alpha * it->second;
	}
}

void IUpdateRule::init_structures()
{
	_init_structures = true;

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_update[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
