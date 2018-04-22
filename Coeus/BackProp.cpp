#include "BackProp.h"

using namespace Coeus;

BackProp::BackProp(NeuralNetwork* p_network)
{
	_network = p_network;
	_cost_function = nullptr;
	_network_gradient = nullptr;
}


BackProp::~BackProp()
{
	delete _cost_function;
	delete _network_gradient;
}

void BackProp::init(ICostFunction* p_cost_function) {
	_cost_function = p_cost_function;

	_network_gradient = new NetworkGradient(_network, _cost_function);
}

double BackProp::train(Tensor* p_input, Tensor* p_target) {
	_network->activate(p_input);

	const double error = _cost_function->cost(_network->get_output(), p_target);
	
	_network_gradient->calc_gradient(p_target);

	calc_update();

	return error;
}

void BackProp::calc_update() {

	for(auto it = _network_gradient->get_gradient()->begin(); it != _network_gradient->get_gradient()->end(); ++it ) {
		_update[it->first] = it->second;
	}

}
