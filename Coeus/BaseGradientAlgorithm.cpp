#include "BaseGradientAlgorithm.h"

using namespace Coeus;

BaseGradientAlgorithm::BaseGradientAlgorithm(NeuralNetwork* p_network)
{
	_network = p_network;
	_cost_function = nullptr;
	_network_gradient = nullptr;
	_alpha = 0;
}


BaseGradientAlgorithm::~BaseGradientAlgorithm()
{
	delete _cost_function;
	delete _network_gradient;
}

double BaseGradientAlgorithm::train(Tensor* p_input, Tensor* p_target) {
	_network->activate(p_input);
	
	const double error = _cost_function->cost(_network->get_output(), p_target);
	
	_network_gradient->calc_gradient(p_target);

	calc_update();
	_network_gradient->update(_update);

	return error;
}

void BaseGradientAlgorithm::init(ICostFunction* p_cost_function, const double p_alpha) {
	_cost_function = p_cost_function;
	_network_gradient = new NetworkGradient(_network, _cost_function);
	_alpha = p_alpha;
	_init_structures = false;
}

void BaseGradientAlgorithm::calc_update() {
	if (!_init_structures) {
		init_structures();
	}

	for (auto it = _network_gradient->get_b_gradient()->begin(); it != _network_gradient->get_b_gradient()->end(); ++it) {
		_update[it->first] = -_alpha * it->second;
	}
}

void BaseGradientAlgorithm::init_structures() {
	_init_structures = true;

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {
		_update[it->first] = Tensor(it->second.rank(), it->second.shape(), Tensor::INIT::ZERO);
	}
}
