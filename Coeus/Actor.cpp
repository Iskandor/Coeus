#include "Actor.h"
#include "RuleFactory.h"

using namespace Coeus;

Actor::Actor(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm, const float p_beta)
{
	_network = p_network;
	_network_gradient = nullptr;
	_gradient_algorithm = p_gradient_algorithm;
	_update_rule = nullptr;
	_beta = p_beta;
}

Actor::Actor(NeuralNetwork* p_network, const GRADIENT_RULE p_grad_rule, const float p_alpha) {
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_gradient_algorithm = nullptr;
	_update_rule = new ActorRule(_network_gradient, RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha), p_alpha);
	_beta = 0;
}

Actor::~Actor()
{
	delete _network_gradient;
	delete _update_rule;

}

float Actor::train(Tensor* p_state0, const int p_action, const float p_td_error) const
{
	_network->activate(p_state0);

	if (_update_rule != nullptr)
	{
		
		Tensor mask = Tensor::Zero({ _network->get_output()->size() });
		mask[p_action] = 1;

		_network_gradient->calc_gradient(&mask);
		_update_rule->calc_update(_network_gradient->get_gradient(), p_td_error, _network->get_output());
		_network->update(_update_rule->get_update());
	}
	if (_gradient_algorithm != nullptr)
	{
		Tensor target = *_network->get_output();

		target[p_action] = target[p_action] + _beta  * p_td_error;

		_gradient_algorithm->train(p_state0, &target);
	}

	return p_td_error;
}
