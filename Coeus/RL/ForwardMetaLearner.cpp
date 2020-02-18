#include "ForwardMetaLearner.h"
#include "RuleFactory.h"
#include "NeuronOperator.h"
#include "QuadraticCost.h"

using namespace Coeus;

ForwardMetaLearner::ForwardMetaLearner(ForwardModel* p_forward_model, NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha) :
	_forward_model(p_forward_model),
	_network(p_network),
	_input(nullptr)
{
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_rule, p_network, p_alpha);
}

ForwardMetaLearner::~ForwardMetaLearner()
{
	delete _network_gradient;
	delete _update_rule;
	delete _input;
}

float ForwardMetaLearner::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1)
{
	QuadraticCost cost_function;

	const float forward_model_error = _forward_model->train(p_state0, p_action, p_state1);
	Tensor fme({ 1 }, Tensor::VALUE, forward_model_error);

	activate(p_state0, p_action);

	const float metalearner_error = cost_function.cost(_network->get_output(), &fme);
	Tensor loss = cost_function.cost_deriv(_network->get_output(), &fme);
	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient());
	_network->update(_update_rule->get_update());

	return metalearner_error;
}

void ForwardMetaLearner::activate(Tensor* p_state, Tensor* p_action)
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
