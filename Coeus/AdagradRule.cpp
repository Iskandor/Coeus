#include "AdagradRule.h"

using namespace Coeus;

AdagradRule::AdagradRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_epsilon): IUpdateRule(p_network_gradient, p_alpha), _epsilon(p_epsilon)
{
}

AdagradRule::~AdagradRule()
= default;

void AdagradRule::calc_update() {
	IUpdateRule::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		Tensor* G = &_G[it->first];
		Tensor* update = &_update[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*G)[i] = (*G)[i] + pow(it->second[i], 2);
			(*update)[i] = -_alpha / sqrt((*G)[i] + _epsilon) * it->second[i];
		}
	}
}

void AdagradRule::init_structures() {
	IUpdateRule::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_G[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
