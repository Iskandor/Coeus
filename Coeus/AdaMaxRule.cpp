#include "AdaMaxRule.h"

using namespace Coeus;

AdaMaxRule::AdaMaxRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon): 
			IUpdateRule(p_network_gradient, p_alpha), _beta1(p_beta1), _beta2(p_beta2), _epsilon(p_epsilon)
{
}

AdaMaxRule::~AdaMaxRule()
= default;

void AdaMaxRule::update_momentum(const string& p_id, Tensor& p_gradient) {

	Tensor *m = &_m[p_id];
	for (int i = 0; i < p_gradient.size(); i++) {
		(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * p_gradient[i];
	}

	Tensor *u = &_u[p_id];
	for (int i = 0; i < p_gradient.size(); i++) {
		(*u)[i] = max(_beta2 * (*u)[i], abs(p_gradient[i]));
	}

	Tensor *m_mean = &_m_mean[p_id];
	for (int i = 0; i < p_gradient.size(); i++) {
		(*m_mean)[i] = (*m)[i] / (1 - _beta1);
	}

}

void AdaMaxRule::calc_update() {
	IUpdateRule::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* u = &_u[it->first];
		Tensor* m_mean = &_m_mean[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / ((*u)[i] + _epsilon) * (*m_mean)[i];
		}
	}
}

void AdaMaxRule::init_structures() {
	IUpdateRule::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_m[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_m_mean[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_u[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
