#include "Adagrad.h"

using namespace Coeus;

Adagrad::Adagrad(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network)
{
	_epsilon = 0;
}


Adagrad::~Adagrad()
{
}

void Adagrad::init(ICostFunction* p_cost_function, const double p_alpha, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);
	_epsilon = p_epsilon;
}

void Adagrad::calc_update() {
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		Tensor* G = &_G[it->first];
		Tensor* update = &_update[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*G)[i] = (*G)[i] + pow(it->second[i], 2);
			(*update)[i] = -_alpha / sqrt((*G)[i] + _epsilon) * it->second[i];
		}
	}
}

void Adagrad::init_structures() {
	BaseGradientAlgorithm::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_G[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
