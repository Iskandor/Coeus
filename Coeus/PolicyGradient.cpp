#include "PolicyGradient.h"
#include "ADAMRule.h"
#include "RuleFactory.h"

using namespace Coeus;

PolicyGradient::PolicyGradient(NeuralNetwork* p_network)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
}

PolicyGradient::~PolicyGradient()
{
	delete _network_gradient;
}

map<string, Tensor>& PolicyGradient::get_gradient(Tensor* p_state, const int p_action, const float p_delta) const
{
	_network->activate(p_state);
	Tensor loss({ _network->get_output_dim() }, Tensor::ZERO);
	loss[p_action] = - p_delta / _network->get_output()->at(p_action);

	_network_gradient->calc_gradient(&loss);

	return _network_gradient->get_gradient();
}
