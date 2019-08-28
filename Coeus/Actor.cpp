#include "Actor.h"
#include "ADAMRule.h"
#include "RuleFactory.h"

using namespace Coeus;

Actor::Actor(NeuralNetwork* p_network, const GRADIENT_RULE p_rule, const float p_alpha)
{
	_network = p_network;
	_alpha = p_alpha;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_rule, _network_gradient, p_alpha);
}


Actor::~Actor()
{
	delete _update_rule;
	delete _network_gradient;
}

void Actor::train(Tensor* p_state, const int p_action, const float p_delta) const
{
	_network->activate(p_state);
	Tensor loss({ _network->get_output_dim() }, Tensor::ZERO);
	loss[p_action] = - p_delta /  _network->get_output()->at(p_action);

	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient(), _alpha);
	_network->update(_update_rule->get_update());
}
