#include "PowerSignRule.h"

using namespace Coeus;

PowerSignRule::PowerSignRule(ParamModel* p_model, const float p_alpha) : IUpdateRule(p_model, p_alpha)
{
	_m = p_model->get_empty_params();
}

PowerSignRule::~PowerSignRule()
= default;

void PowerSignRule::calc_update(Gradient& p_gradient, const float p_alpha)
{
	const float beta1 = 0.9f;
	IUpdateRule::calc_update(p_gradient, p_alpha);

	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {

		Tensor* m = &_m[it->first];
		Tensor* g = &it->second;
		Tensor* update = &_update[it->first];

		for (int i = 0; i < g->size(); i++) {
			(*m)[i] = beta1 * (*m)[i] + (1 - beta1) * (*g)[i];
			(*update)[i] = -pow(_alpha, Tensor::sign((*m)[i] * Tensor::sign((*g)[i]))) * (*g)[i];
		}
	}
}

IUpdateRule* PowerSignRule::clone(ParamModel* p_model)
{
	return new PowerSignRule(p_model, _alpha);
}