#include "CACLA.h"
#include "RuleFactory.h"
#include "RandomGenerator.h"
#include "QuadraticCost.h"

using namespace Coeus;

CACLA::CACLA(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha, float p_beta) : 
	_network(p_network),
	_alpha(p_alpha),
	_beta(p_beta),
	_var(1)
{
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_rule, p_network, p_alpha);
}


CACLA::~CACLA()
{
	delete _update_rule;
	delete _network_gradient;
}

void CACLA::train(Tensor* p_state, Tensor* p_action, float p_delta)
{
	if (_beta > 0) _var = (1 - _beta) * _var + _beta * p_delta * p_delta;

	if (p_delta > 0)
	{		
		int v = ceil(p_delta / sqrt(_var));
		_network->activate(p_state);

		Tensor loss = _mse.cost_deriv(_network->get_output(), p_action);

		_network_gradient->calc_gradient(&loss);
		_update_rule->calc_update(_network_gradient->get_gradient(), _alpha * v);
		_network->update(_update_rule->get_update());
	}
}

Tensor CACLA::get_action(Tensor* p_state, float p_sigma) const
{
	Tensor output({ _network->get_output_dim() }, Tensor::ZERO);
	_network->activate(p_state);

	for(int i = 0; i < _network->get_output_dim(); i++)
	{
		float rand = RandomGenerator::get_instance().normal_random(0, p_sigma);
		output[i] = _network->get_output()->at(i) + rand;
		//output[i] = RandomGenerator::get_instance().normal_random(0, p_sigma);
	}

	return output;
}
