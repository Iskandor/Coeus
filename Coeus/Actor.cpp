#include "Actor.h"
#include "RuleFactory.h"

using namespace Coeus;

Actor::Actor(NeuralNetwork* p_network, const GRADIENT_RULE p_grad_rule, const double p_alpha) {
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new ActorRule(_network_gradient, RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha), p_alpha);

}

Actor::~Actor()
{
	delete _network_gradient;
	delete _update_rule;

}

double Actor::train(Tensor* p_state0, const int p_action, const double p_td_error) const
{
	_network->activate(p_state0);
	Tensor mask = Tensor::Zero({ _network->get_output()->size() });
	mask[p_action] = 1;

	_network_gradient->calc_gradient(&mask);
	_update_rule->calc_update(_network_gradient->get_gradient(), p_td_error, _network->get_output());
	_network->update(_update_rule->get_update());

	return p_td_error;
}
