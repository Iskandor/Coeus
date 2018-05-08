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

void RMSProp::update_cache(const string p_id, Tensor& p_gradient) {
	if (_cache.find(p_id) != _cache.end()) {
		for (int i = 0; i < p_gradient.size(); i++) {
			_cache[p_id][i] = _decay * _cache[p_id][i] + (1 - _decay) * pow(p_gradient[i], 2);
		}
	}
	else {
		_cache[p_id] = Tensor(p_gradient);
		for (int i = 0; i < p_gradient.size(); i++) {
			_cache[p_id][i] = (1 - _decay) * pow(p_gradient[i], 2);
		}
	}
}


void RMSProp::calc_update() {
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		update_cache(it->first, it->second);

		if (_update.find(it->first) == _update.end()) {
			_update[it->first] = Tensor(it->second);
		}

		for(int i = 0; i < it->second.size(); i++) {
			_update[it->first][i] = -_alpha / sqrt(_cache[it->first][i] + _epsilon) * it->second[i];
		}
	}
}
