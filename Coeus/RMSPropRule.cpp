#include "RMSPropRule.h"

using namespace Coeus;

RMSPropRule::RMSPropRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_decay, const double p_epsilon):
	IUpdateRule(p_network_gradient, p_alpha), _decay(p_decay), _epsilon(p_epsilon)
{
}

RMSPropRule::~RMSPropRule()
= default;

void RMSPropRule::calc_update() {
	IUpdateRule::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		/*
		for(int i = 0; i < it->second.size(); i++) {
		_cache[it->first][i] = _decay * _cache[it->first][i] + (1 - _decay) * pow(it->second[i], 2);
		_update[it->first][i] = -_alpha / sqrt(_cache[it->first][i] + _epsilon) * it->second[i];
		}
		*/

		_cache[it->first] = _decay * _cache[it->first] + (1 - _decay) * it->second.pow(2);
		_update[it->first] = (-_alpha / (_cache[it->first] + _epsilon).sqrt()).dot(it->second);
	}
}

void RMSPropRule::init_structures() {
	IUpdateRule::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_cache[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
