#include "CACLA.h"
#include "RuleFactory.h"
#include "RandomGenerator.h"
#include "QuadraticCost.h"

using namespace Coeus;

CACLA::CACLA(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha)
{
	_network = p_network;
	_alpha = p_alpha;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_rule, _network_gradient, p_alpha);
}


CACLA::~CACLA()
{
	delete _update_rule;
	delete _network_gradient;
}

void CACLA::train(Tensor* p_state, Tensor* p_action, float p_delta)
{
	if (p_delta > 0)
	{		
		_network->activate(p_state);

		Tensor loss = _mse.cost_deriv(_network->get_output(), p_action);

		_network_gradient->calc_gradient(&loss);
		_update_rule->calc_update(_network_gradient->get_gradient(), _alpha);
		_network->update(_update_rule->get_update());
	}
}

Tensor CACLA::get_action(Tensor* p_state, float p_sigma) const
{
	Tensor output({ _network->get_output_dim() }, Tensor::ZERO);
	_network->activate(p_state);

	for(int i = 0; i < _network->get_output_dim(); i++)
	{
		output[i] = RandomGenerator::get_instance().normal_random(_network->get_output()->at(i), p_sigma);
	}

	return output;
}
