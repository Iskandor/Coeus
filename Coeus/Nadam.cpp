#include "Nadam.h"

using namespace Coeus;

Nadam::Nadam(NeuralNetwork * p_network) : BaseGradientAlgorithm(p_network)
{
	_beta1 = 0;
	_beta2 = 0;
	_epsilon = 0;
}

Nadam::~Nadam()
{
}

void Nadam::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);
	_beta1 = p_beta1;
	_beta2 = p_beta2;
	_epsilon = p_epsilon;
}

void Coeus::Nadam::update_momentum(string p_id, Tensor & p_gradient)
{
	if (_momentum1.find(p_id) == _momentum1.end()) {
		_momentum1[p_id] = Tensor(p_gradient);
		_momentum1_est[p_id] = Tensor(p_gradient);
		for (int i = 0; i < p_gradient.size(); i++) {
			_momentum1[p_id][i] = (1 - _beta1) * p_gradient[i];
		}
	}
	else {
		for (int i = 0; i < p_gradient.size(); i++) {
			_momentum1[p_id][i] = _beta1 * _momentum1[p_id][i] + (1 - _beta1) * p_gradient[i];
		}
	}

	if (_momentum2.find(p_id) == _momentum2.end()) {
		_momentum2[p_id] = Tensor(p_gradient);
		_momentum2_est[p_id] = Tensor(p_gradient);
		for (int i = 0; i < p_gradient.size(); i++) {
			_momentum2[p_id][i] = (1 - _beta2) * pow(p_gradient[i], 2);
		}
	}
	else {
		for (int i = 0; i < p_gradient.size(); i++) {
			_momentum2[p_id][i] = _beta2 * _momentum2[p_id][i] + (1 - _beta2) * pow(p_gradient[i], 2);
		}
	}

	for (int i = 0; i < p_gradient.size(); i++) {
		_momentum1_est[p_id][i] = _momentum1[p_id][i] / (1 - _beta1);
		_momentum2_est[p_id][i] = _momentum2[p_id][i] / (1 - _beta2);
	}
}

void Coeus::Nadam::calc_update()
{
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		if (_update.find(it->first) == _update.end()) {
			_update[it->first] = Tensor(it->second);
		}

		update_momentum(it->first, it->second);

		Tensor* _u = &_update[it->first];
		Tensor* _g = &it->second;
		Tensor* _m_mean = &_momentum1_est[it->first];
		Tensor* _v_mean = &_momentum2_est[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*_u)[i] = -_alpha / (sqrt((*_v_mean)[i]) + _epsilon) * (_beta1 * (*_m_mean)[i] + (1 - _beta1) * (*_g)[i] / (1 - _beta1));
		}
	}
}
