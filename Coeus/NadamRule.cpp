#include "NadamRule.h"

using namespace Coeus;


NadamRule::NadamRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_beta1, const double p_beta2,
                     const double p_epsilon): IUpdateRule(p_network_gradient, p_alpha), _beta1(p_beta1), _beta2(p_beta2), _epsilon(p_epsilon)
{
}

NadamRule::~NadamRule()
= default;

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

void NadamRule::calc_update()
{
	IUpdateRule::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (_beta1 * (*m_mean)[i] + (1 - _beta1) * it->second[i] / (1 - _beta1));
		}
	}
}

void NadamRule::init_structures() {
	IUpdateRule::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_m[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_v[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_m_mean[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_v_mean[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}