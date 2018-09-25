#include "RMSProp.h"

using namespace Coeus;

RMSProp::RMSProp(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network)
{
	_decay = 0;
	_epsilon = 0;
}


RMSProp::~RMSProp()
{
}

void RMSProp::init(ICostFunction* p_cost_function, const double p_alpha, const double p_decay, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);

	_decay = p_decay;
	_epsilon = p_epsilon;
}

void RMSProp::calc_update() {
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		/*
		for(int i = 0; i < it->second.size(); i++) {
			_cache[it->first][i] = _decay * _cache[it->first][i] + (1 - _decay) * pow(it->second[i], 2);
			_update[it->first][i] = -_alpha / sqrt(_cache[it->first][i] + _epsilon) * it->second[i];
		}
		*/

		_cache[it->first] = _decay * _cache[it->first] + (1 - _decay) * it->second.pow(2);
		_update[it->first] = (-_alpha / (_cache[it->first] + _epsilon).sqrt()).dot(it->second);
	}
}

void RMSProp::init_structures() {
	BaseGradientAlgorithm::init_structures();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_cache[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
