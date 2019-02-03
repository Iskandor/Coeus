#include "RMSPropRule.h"

using namespace Coeus;

RMSPropRule::RMSPropRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_decay, const double p_epsilon):
	IUpdateRule(p_network_gradient, p_alpha), _decay(p_decay), _epsilon(p_epsilon)
{
	_cache = p_network_gradient->get_empty_params();
}

RMSPropRule::~RMSPropRule()
= default;

void RMSPropRule::calc_update(map<string, Tensor>* p_gradient, const double p_alpha) {
	IUpdateRule::calc_update(p_gradient, p_alpha);
	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {

		Tensor* cache = &_cache[it->first];
		Tensor* update = &_update[it->first];

		for(int i = 0; i < it->second.size(); i++) {
			(*cache)[i] = _decay * (*cache)[i] + (1 - _decay) * pow(it->second[i], 2);
			(*update)[i] = -_alpha / sqrt((*cache)[i] + _epsilon) * it->second[i];
		}
	}
}

IUpdateRule* RMSPropRule::clone(NetworkGradient* p_network_gradient)
{
	return new RMSPropRule(p_network_gradient, _alpha, _decay, _epsilon);
}

void RMSPropRule::reset()
{
	for (auto it = _update.begin(); it != _update.end(); ++it) {
		_cache[it->first].fill(0);
	}
}
