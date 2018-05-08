#include "Adadelta.h"

using namespace Coeus;

Adadelta::Adadelta(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network) {
	_epsilon = 0;
	_decay = 0;
}


Adadelta::~Adadelta()
{
}

void Adadelta::init(ICostFunction* p_cost_function, const double p_alpha, const double p_decay, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, p_alpha);
	_decay = p_decay;
	_epsilon = p_epsilon;
}

void Adadelta::update_cache(const string p_id, Tensor& p_gradient) {
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

	if (_cache_delta.find(p_id) == _cache_delta.end()) {
		_cache_delta[p_id] = Tensor(p_gradient);
		_cache_delta[p_id].fill(0);
	}
}

void Adadelta::update_cache_delta(const string p_id, Tensor& p_gradient) {
	for (int i = 0; i < p_gradient.size(); i++) {
		_cache_delta[p_id][i] = _decay * _cache_delta[p_id][i] + (1 - _decay) * pow(_alpha / sqrt(_cache[p_id][i] + _epsilon) * p_gradient[i], 2);
	}
}

void Adadelta::calc_update() {
	BaseGradientAlgorithm::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		if (_update.find(it->first) == _update.end()) {
			_update[it->first] = Tensor(it->second);
		}

		update_cache(it->first, it->second);

		for (int i = 0; i < it->second.size(); i++) {
			_update[it->first][i] = -sqrt(_cache_delta[it->first][i] + _epsilon) / sqrt(_cache[it->first][i] + _epsilon) * it->second[i];
		}

		update_cache_delta(it->first, it->second);
	}
}

