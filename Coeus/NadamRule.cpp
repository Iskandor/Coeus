#include "NadamRule.h"

using namespace Coeus;


NadamRule::NadamRule(NetworkGradient* p_network_gradient, const float p_alpha, const float p_beta1, const float p_beta2,
                     const float p_epsilon): IUpdateRule(p_network_gradient, p_alpha), _beta1(p_beta1), _beta2(p_beta2), _epsilon(p_epsilon)
{
	_m = p_network_gradient->get_empty_params();
	_m_mean = p_network_gradient->get_empty_params();
	_v = p_network_gradient->get_empty_params();
	_v_mean = p_network_gradient->get_empty_params();
}

NadamRule::~NadamRule()
= default;

IUpdateRule* NadamRule::clone(NetworkGradient* p_network_gradient)
{
	return new NadamRule(p_network_gradient, _alpha, _beta1, _beta2, _epsilon);
}

void NadamRule::reset()
{
	for (auto key = _update.begin(); key != _update.end(); ++key)
	{
		_m[key->first].fill(0);
		_v[key->first].fill(0);
		_m_mean[key->first].fill(0);
		_v_mean[key->first].fill(0);
	}
}

void NadamRule::update_momentum(const string& p_id, Tensor & p_gradient)
{
	Tensor* m = &_m[p_id];
	Tensor* v = &_v[p_id];
	Tensor* m_mean = &_m_mean[p_id];
	Tensor* v_mean = &_v_mean[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * p_gradient[i];
		(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow(p_gradient[i], 2);
		(*m_mean)[i] = (*m)[i] / (1 - _beta1);
		(*v_mean)[i] = (*v)[i] / (1 - _beta2);
	}
}

void NadamRule::calc_update(map<string, Tensor>* p_gradient, const float p_alpha)
{
	IUpdateRule::calc_update(p_gradient, p_alpha);

	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {

		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (_beta1 * (*m_mean)[i] + (1 - _beta1) * it->second[i] / (1 - _beta1));
		}
	}
}