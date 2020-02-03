#include "ForwardModel.h"
#include "RuleFactory.h"
#include "NeuronOperator.h"
#include "QuadraticCost.h"

using namespace Coeus;

ForwardModel::ForwardModel(NeuralNetwork* p_network, const GRADIENT_RULE p_rule, const float p_alpha):
	_network(p_network),
	_input(nullptr)
{
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_rule, p_network, p_alpha);
}

ForwardModel::~ForwardModel()
{
	delete _network_gradient;
	delete _update_rule;
	delete _input;
}

float ForwardModel::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1)
{
	QuadraticCost cost_function;
	
	activate(p_state0, p_action);

	const float error = cost_function.cost(_network->get_output(), p_state1);
	Tensor loss = cost_function.cost_deriv(_network->get_output(), p_state1);	
	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient());
	_network->update(_update_rule->get_update());

	return error;
}

void ForwardModel::activate(Tensor* p_state, Tensor* p_action)
{
	if (p_state->rank() == 2 && p_action->rank() == 2)
	{
		_input = NeuronOperator::init_auxiliary_parameter(_input, p_state->shape(0), _network->get_input_dim());
	}
	if (p_state->rank() == 1 && p_action->rank() == 1)
	{
		_input = NeuronOperator::init_auxiliary_parameter(_input, 1, _network->get_input_dim());
		_input->reset_index();
		_input->push_back(p_state);
		_input->push_back(p_action);
	}

	_network->activate(_input);
}
