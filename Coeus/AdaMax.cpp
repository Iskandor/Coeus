#include "AdaMax.h"

using namespace Coeus;

AdaMax::AdaMax(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network) {
	_beta1 = 0;
	_beta2 = 0;
	_epsilon = 0;
}

AdaMax::~AdaMax()
{
}

void AdaMax::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);
	_beta1 = p_beta1;
	_beta2 = p_beta2;
	_epsilon = p_epsilon;
}

void AdaMax::update_momentum(const string p_id, Tensor& p_gradient) {
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

	if (_inf_norm.find(p_id) == _inf_norm.end()) {
		_inf_norm[p_id] = Tensor(p_gradient);
		for (int i = 0; i < p_gradient.size(); i++) {
			_inf_norm[p_id][i] = abs(p_gradient[i]);
		}
	}
	else {
		for (int i = 0; i < p_gradient.size(); i++) {
			_inf_norm[p_id][i] = max(_beta2 * _inf_norm[p_id][i], abs(p_gradient[i]));
		}
	}

	for (int i = 0; i < p_gradient.size(); i++) {
		_momentum1_est[p_id][i] = _momentum1[p_id][i] / (1 - _beta1);
	}

}

void AdaMax::calc_update() {
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		if (_update.find(it->first) == _update.end()) {
			_update[it->first] = Tensor(it->second);
		}

		update_momentum(it->first, it->second);

		for (int i = 0; i < it->second.size(); i++) {
			_update[it->first][i] = -_alpha / (_inf_norm[it->first][i] + _epsilon) * _momentum1_est[it->first][i];
		}
	}
}

