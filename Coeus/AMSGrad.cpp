#include "AMSGrad.h"

using namespace Coeus;

AMSGrad::AMSGrad(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network)
{
	_beta1 = 0;
	_beta2 = 0;
	_epsilon = 0;
}


AMSGrad::~AMSGrad()
{
}

void AMSGrad::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);
	_beta1 = p_beta1;
	_beta2 = p_beta2;
	_epsilon = p_epsilon;
}

void AMSGrad::update_momentum(const string p_id, Tensor& p_gradient) {

	Tensor* m = &_m[p_id];
	Tensor* v = &_v[p_id];
	Tensor* v_mean = &_v_mean[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * p_gradient[i];
		(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow(p_gradient[i], 2);
		(*v_mean)[i] = max((*v_mean)[i], (*v)[i]);
	}

}

void AMSGrad::calc_update() {
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m = &_m[it->first];
		Tensor* v_mean = &_v_mean[it->first];


		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (*m)[i];
		}
	}
}

void AMSGrad::init_structures() {
	BaseGradientAlgorithm::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_m[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_v[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
		_v_mean[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
