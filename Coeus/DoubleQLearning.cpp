#include "RandomGenerator.h"
#include "DoubleQLearning.h"
#include "RuleFactory.h"

using namespace Coeus;

DoubleQLearning::DoubleQLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda):
	_alpha(p_alpha),
	_gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha);

	_action_count = p_network->get_output_dim() / 2;
	_output = new Tensor({ _action_count }, Tensor::ZERO);
}

DoubleQLearning::~DoubleQLearning()
{
	delete _output;
	delete _network_gradient;
	delete _update_rule;
}

float DoubleQLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const float p_reward) const
{

	float Qs0a = 0;
	float maxQs1a = 0;
	Tensor loss({ _network->get_output_dim() }, Tensor::ZERO);

	_network->activate(p_state0);

	if (RandomGenerator::get_instance().random() < 0.5) {		
		Qs0a = _network->get_output()->at(p_action0);
		maxQs1a = calc_max_qa(p_state1, _network, 0);
		const float delta = p_reward + _gamma * maxQs1a - Qs0a;
		loss[p_action0] = Qs0a - delta;
	}
	else
	{
		Qs0a = _network->get_output()->at(p_action0 + _action_count);
		maxQs1a = calc_max_qa(p_state1, _network, _action_count);
		const float delta = p_reward + _gamma * maxQs1a - Qs0a;
		loss[p_action0] = Qs0a - delta;
	}

	_network->activate(p_state0);
	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient(), _alpha);
	_network->update(_update_rule->get_update());

	return Qs0a;
}

Tensor* DoubleQLearning::get_output() const
{
	for(int i = 0; i < _action_count; i++)
	{
		(*_output)[i] = (_network->get_output()->at(i) + _network->get_output()->at(i + _action_count)) / 2;
	}

	return _output;
}

float DoubleQLearning::calc_max_qa(Tensor* p_state, NeuralNetwork* p_network, const int p_index) const {
	p_network->activate(p_state);
	int maxQa = p_index;

	for(int i = 0; i < _action_count; i++)
	{
		if (p_network->get_output()->at(i + p_index) > p_network->get_output()->at(maxQa))
		{
			maxQa = i + p_index;
		}
	}

	return p_network->get_output()->at(maxQa + (_action_count - p_index));
}
