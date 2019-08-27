#include "Reinforce.h"
#include "TensorOperator.h"
#include "ADAMRule.h"

using namespace Coeus;

Reinforce::Reinforce(NeuralNetwork* p_network, const float p_alpha)
{
	_network = p_network;
	_alpha = p_alpha;
	_t = 1;

	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new ADAMRule(_network_gradient, p_alpha);
	_update_rule->set_step(_t);
	//_update_rule = new ReinforceRule(_network_gradient, p_alpha);
}


Reinforce::~Reinforce()
{
	delete _update_rule;
	delete _network_gradient;
}

void Reinforce::train(Tensor* p_state, const int p_action, const float p_delta)
{
	_network->activate(p_state);
	Tensor loss({ _network->get_output_dim() }, Tensor::ZERO);
	loss[p_action] = - p_delta /  _network->get_output()->at(p_action);

	_network_gradient->calc_gradient(&loss);
	map<string, Tensor> *gradient = _network_gradient->get_gradient();
	_update_rule->calc_update(gradient, _alpha);
	_t++;
	//_update_rule->calc_update(gradient, 1, -_alpha);
	_network->update(_update_rule->get_update());
	_network->activate(p_state);
}
