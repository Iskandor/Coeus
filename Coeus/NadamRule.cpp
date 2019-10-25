#include "NadamRule.h"

using namespace Coeus;


NadamRule::NadamRule(NetworkGradient* p_network_gradient, const float p_alpha, const float p_beta1, const float p_beta2,
                     const float p_epsilon): IUpdateRule(p_network_gradient, p_alpha), _beta1(p_beta1), _beta2(p_beta2), _epsilon(p_epsilon)
{
	_m = p_network_gradient->get_network()->get_empty_params();
	_m_mean = p_network_gradient->get_network()->get_empty_params();
	_v = p_network_gradient->get_network()->get_empty_params();
	_v_mean = p_network_gradient->get_network()->get_empty_params();
}

NadamRule::~NadamRule()
= default;

IUpdateRule* NadamRule::clone(NetworkGradient* p_network_gradient)
{
	return new NadamRule(p_network_gradient, _alpha, _beta1, _beta2, _epsilon);
}

void NadamRule::update_momentum(const string& p_id, Tensor & p_gradient)
{
	float* gx = &p_gradient.arr()[0];
	float* mx = &_m[p_id].arr()[0];
	float* vx = &_v[p_id].arr()[0];
	float* mmx = &_m_mean[p_id].arr()[0];
	float* vmx = &_v_mean[p_id].arr()[0];

	for (int i = 0; i < p_gradient.size(); i++) {
		*mx = _beta1 * *mx + (1 - _beta1) * *gx;
		*vx = _beta2 * *vx + (1 - _beta2) * pow(*gx, 2);
		*mmx++ = *mx++ / (1 - _beta1);
		*vmx++ = *vx++ / (1 - _beta2);
		gx++;
	}
}

void NadamRule::calc_update(map<string, Tensor>& p_gradient, const float p_alpha)
{
	IUpdateRule::calc_update(p_gradient, p_alpha);

	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {

		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (_beta1 * (*m_mean)[i] + (1 - _beta1) * it->second[i] / (1 - _beta1));
		}
	}
}