#include "BackProph.h"

using namespace Coeus;

BackProp::BackProp(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network) {
	_momentum = 0;
	_nesterov = false;
}

BackProp::~BackProp() {
	
}

void BackProp::init(ICostFunction* p_cost_function, const double p_alpha, const double p_momentum, const bool p_nesterov) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);

	_momentum = p_momentum;
	_nesterov = p_nesterov;
}

void BackProp::calc_update() {
	BaseGradientAlgorithm::calc_update();

	Tensor prev_update;

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		if (_nesterov) {
			prev_update = Tensor(_update[it->first]);
		}

		if (_momentum > 0) {
			Tensor* update = &_update[it->first];

			for (int i = 0; i < it->second.size(); i++) {
				(*update)[i] = _momentum * (*update)[i] - _alpha * it->second[i];
			}

			if(_nesterov) {
				for (int i = 0; i < it->second.size(); i++) {
					(*update)[i] = -_momentum * prev_update[i] + (1 + _momentum) * (*update)[i];
				}
			}
		}
	}
}

