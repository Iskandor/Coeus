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

		if (_G.find(it->first) == _G.end()) {
			_G[it->first] = Tensor(it->second);

			for (int i = 0; i < it->second.size(); i++) {
				_G[it->first][i] = pow(it->second[i], 2);
			}
		}
		else {
			for (int i = 0; i < it->second.size(); i++) {
				_G[it->first][i] = _G[it->first][i] + pow(it->second[i], 2);
			}
		}

		if (_update.find(it->first) == _update.end()) {
			_update[it->first] = Tensor(it->second);
		}

		for (int i = 0; i < it->second.size(); i++) {
			
			_update[it->first][i] = -_alpha / sqrt(_G[it->first][i] + _epsilon) * it->second[i];
		}
	}
}
