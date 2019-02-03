#include "TD.h"
#include "RuleFactory.h"

using namespace Coeus;

TD::TD(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, double p_alpha, double p_gamma, double p_lambda) :
	_gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new GeneralTDRule(_network_gradient, RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha), p_alpha, p_gamma, p_lambda);
}

TD::~TD()
{
	delete _network_gradient;
	delete _update_rule;
}

double TD::train(Tensor* p_state0, Tensor* p_state1, const double p_reward) const
{
	_network->activate(p_state0);
	const double Vs0 = _network->get_output()->at(0);
	_network->activate(p_state1);
	const double Vs1 = _network->get_output()->at(0);

	const double delta = p_reward + _gamma * Vs1 - Vs0;

	_network_gradient->calc_gradient();
	_update_rule->calc_update(_network_gradient->get_gradient(), delta, 0);
	_network->update(_update_rule->get_update());

	return delta;
}
