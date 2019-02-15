#include "PowerSignRule.h"
#include "FLAB.h"

using namespace Coeus;

PowerSignRule::PowerSignRule(NetworkGradient* p_network_gradient, const float p_alpha) : IUpdateRule(p_network_gradient, p_alpha)
{
	_m = p_network_gradient->get_empty_params();
}

PowerSignRule::~PowerSignRule()
= default;

void PowerSignRule::calc_update(map<string, Tensor>* p_gradient, const float p_alpha)
{
	const float beta1 = 0.9;
	IUpdateRule::calc_update(p_gradient, p_alpha);

	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {

		Tensor* m = &_m[it->first];
		Tensor* g = &it->second;
		Tensor* update = &_update[it->first];

		for (int i = 0; i < g->size(); i++) {
			(*m)[i] = beta1 * (*m)[i] + (1 - beta1) * (*g)[i];
			(*update)[i] = -pow(_alpha, sign((*m)[i] * sign((*g)[i]))) * (*g)[i];
		}
	}
}

IUpdateRule* PowerSignRule::clone(NetworkGradient* p_network_gradient)
{
	return new PowerSignRule(p_network_gradient, _alpha);
}

void PowerSignRule::reset()
{
}
