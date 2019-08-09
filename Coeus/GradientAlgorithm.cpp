#include "GradientAlgorithm.h"
#include <chrono>

using namespace Coeus;

GradientAlgorithm::GradientAlgorithm(NeuralNetwork* p_network)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(_network);
	_cost_function = nullptr;	
	_update_rule = nullptr;
	_recurrent_mode = BPTT;
}


GradientAlgorithm::~GradientAlgorithm()
{
	delete _update_rule;
	delete _cost_function;
	delete _network_gradient;
}

float GradientAlgorithm::train(Tensor* p_input, Tensor* p_target)
{
	float error = 0;
	float alpha = 0;

	_network_gradient->activate(p_input);
	Tensor dloss = _cost_function->cost_deriv(_network->get_output(), p_target);

	_network_gradient->calc_gradient(&dloss);
	error = _cost_function->cost(_network->get_output(), p_target);	
	_update_rule->calc_update(_network_gradient->get_gradient(), alpha);
	_network->update(_update_rule->get_update());

	return error;
}

float GradientAlgorithm::train(vector<Tensor*>* p_input, Tensor* p_target) {

	float error = 0;
	float alpha = 0;

	_network_gradient->set_recurrent_mode(_recurrent_mode);
	_network_gradient->activate(p_input);
	error = _cost_function->cost(_network->get_output(), p_target);
	Tensor dloss = _cost_function->cost_deriv(_network->get_output(), p_target);

	_network_gradient->calc_gradient(p_input, &dloss);
	_update_rule->calc_update(_network_gradient->get_gradient(), alpha);
	_network->update(_update_rule->get_update());

	_network_gradient->set_recurrent_mode(NONE);

	return error;
}

void GradientAlgorithm::reset() const
{
	_network_gradient->reset();
}

void GradientAlgorithm::set_recurrent_mode(const RECURRENT_MODE p_value)
{
	_recurrent_mode = p_value;	
}

void GradientAlgorithm::init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule)
{
	_cost_function = p_cost_function;
	_update_rule = p_update_rule;
}
