#include "TD.h"
#include "RuleFactory.h"

using namespace Coeus;

TD::TD(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda) : 
	_alpha(p_alpha), _gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_grad_rule, p_network, p_alpha);
}

TD::~TD()
{
	delete _network_gradient;
	delete _update_rule;
}

float TD::train(Tensor* p_state0, Tensor* p_state1, const float p_reward, bool p_finished) const
{
	_network->activate(p_state1);
	const float Vs1 = _network->get_output()->at(0);
	_network->activate(p_state0);
	const float Vs0 = _network->get_output()->at(0);

	const float delta = p_finished ? p_reward - Vs0 : p_reward + _gamma * Vs1 - Vs0;

	Tensor loss({ 1 }, Tensor::VALUE, Vs0 - delta);

	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient(), _alpha);
	_network->update(_update_rule->get_update());

	return delta;
}
