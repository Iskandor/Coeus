#include "BackPropRule.h"

using namespace Coeus;

BackPropRule::BackPropRule(NetworkGradient* p_network_gradient, const float p_alpha, const float p_momentum, const bool p_nesterov): IUpdateRule(p_network_gradient, p_alpha),
	_momentum(p_momentum), 
	_nesterov(p_nesterov)
{
}

BackPropRule::~BackPropRule()
= default;

IUpdateRule* BackPropRule::clone(NetworkGradient* p_network_gradient)
{
	return new BackPropRule(p_network_gradient, _alpha, _momentum, _nesterov);
}

void BackPropRule::reset()
{
	for (auto it = _update.begin(); it != _update.end(); ++it) {
		_update[it->first].fill(0);
	}
}

void BackPropRule::calc_update(map<string, Tensor>* p_gradient, const float p_alpha)
{
	IUpdateRule::calc_update(p_gradient, p_alpha);

	Tensor prev_update;

	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		if (_nesterov) {
			prev_update = Tensor(_update[it->first]);
		}

		Tensor* update = &_update[it->first];

		if (_momentum > 0) {
			for (int i = 0; i < it->second.size(); i++) {
				(*update)[i] = _momentum * (*update)[i] - _alpha * it->second[i];
			}

			if (_nesterov) {
				for (int i = 0; i < it->second.size(); i++) {
					(*update)[i] = -_momentum * prev_update[i] + (1 + _momentum) * (*update)[i];
				}
			}
		}
		else {
			for (int i = 0; i < it->second.size(); i++) {
				(*update)[i] = -_alpha * it->second[i];
			}
		}
	}
}
