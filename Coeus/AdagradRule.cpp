#include "AdagradRule.h"

using namespace Coeus;

AdagradRule::AdagradRule(NetworkGradient* p_network_gradient, const float p_alpha, const float p_epsilon): IUpdateRule(p_network_gradient, p_alpha), _epsilon(p_epsilon)
{
	_G = p_network_gradient->get_network()->get_empty_params();
}

AdagradRule::~AdagradRule()
= default;

void AdagradRule::calc_update(map<string, Tensor>* p_gradient, const float p_alpha ) {
	IUpdateRule::calc_update(p_gradient, p_alpha);
	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {

		Tensor* G = &_G[it->first];
		Tensor* update = &_update[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*G)[i] = (*G)[i] + pow(it->second[i], 2);
			(*update)[i] = -_alpha / sqrt((*G)[i] + _epsilon) * it->second[i];
		}
	}
}

IUpdateRule* AdagradRule::clone(NetworkGradient* p_network_gradient)
{
	return new AdagradRule(p_network_gradient, _alpha, _epsilon);
}