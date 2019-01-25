#include "SARSA.h"
#include "RuleFactory.h"

using namespace Coeus;

SARSA::SARSA(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, double p_alpha, double p_gamma, double p_lambda):
	_gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new GeneralTDRule(_network_gradient, RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha), p_alpha, p_gamma, p_lambda);
}

SARSA::~SARSA()
{
	delete _network_gradient;
	delete _update_rule;
}


double SARSA::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const int p_action1, const double p_reward, const bool p_final) const
{
	_network->activate(p_state0);
	const double Qs0a0 = _network->get_output()->at(p_action0);
	_network->activate(p_state1);
	const double Qs1a1 = p_final ? 0 : _network->get_output()->at(p_action1);

	const double delta = p_reward + _gamma * Qs1a1 - Qs0a0;

	Tensor mask = Tensor::Zero({ _network->get_output()->size() });
	mask[p_action0] = 1;

	_network_gradient->calc_gradient(&mask);
	_update_rule->calc_update(_network_gradient->get_gradient(), delta);
	_network->update(_update_rule->get_update());

	return delta;
}

void SARSA::reset_traces() const
{
	_update_rule->reset_traces();
}



