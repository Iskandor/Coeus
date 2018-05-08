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
		if (_nesterov && _update.find(it->first) != _update.end()) {
			prev_update = Tensor(_update[it->first]);
		}

		if (_momentum > 0 && _update.find(it->first) != _update.end()) {
			_update[it->first] = _momentum * _update[it->first] -_alpha * it->second;

			if(_nesterov) {
				_update[it->first] = -_momentum * prev_update + (1 + _momentum) * _update[it->first];
			}
		}
		else {
			_update[it->first] = -_alpha * it->second;
		}		
	}
}

