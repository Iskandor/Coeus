#include "DoubleQLearning.h"
#include "RandomGenerator.h"
#include "DoubleQLearning.h"
#include "RuleFactory.h"

using namespace Coeus;

DoubleQLearning::DoubleQLearning(NeuralNetwork* p_network_a, NeuralNetwork* p_network_b, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda):
	_alpha(p_alpha),
	_gamma(p_gamma)
{
	_network_a = p_network_a;
	_network_gradient_a = new NetworkGradient(p_network_a);
	_update_rule_a = RuleFactory::create_rule(p_grad_rule, _network_gradient_a, p_alpha);
	_network_b = p_network_b;
	_network_gradient_b = new NetworkGradient(p_network_b);
	_update_rule_b = RuleFactory::create_rule(p_grad_rule, _network_gradient_b, p_alpha);

	_action_count = p_network_a->get_output_dim();
	_output = new Tensor({ _action_count }, Tensor::ZERO);
}

DoubleQLearning::~DoubleQLearning()
{
	delete _output;
	delete _network_gradient_a;
	delete _update_rule_a;
	delete _network_gradient_b;
	delete _update_rule_b;
}

float DoubleQLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const float p_reward, bool p_finished) const
{
	float Qs0a = 0;
	float maxQs1a = 0;
	Tensor loss({ _network_a->get_output_dim() }, Tensor::ZERO);

	if (RandomGenerator::get_instance().random() < 0.5) {		
		_network_a->activate(p_state0);
		_network_b->activate(p_state1);
		Qs0a = _network_a->get_output()->at(p_action0);		
		maxQs1a = _network_b->get_output()->at(calc_max_qa_index(p_state1, _network_a));

		const float delta = p_finished ? p_reward - Qs0a : p_reward + _gamma * maxQs1a - Qs0a;
		loss[p_action0] = Qs0a - delta;

		_network_a->activate(p_state0);
		_network_gradient_a->calc_gradient(&loss);
		_update_rule_a->calc_update(_network_gradient_a->get_gradient(), _alpha);
		_network_a->update(_update_rule_a->get_update());
	}
	else
	{
		_network_a->activate(p_state1);
		_network_b->activate(p_state0);
		Qs0a = _network_b->get_output()->at(p_action0);
		maxQs1a = _network_a->get_output()->at(calc_max_qa_index(p_state1, _network_b));

		const float delta = p_finished ? p_reward - Qs0a : p_reward + _gamma * maxQs1a - Qs0a;
		loss[p_action0] = Qs0a - delta;

		_network_b->activate(p_state0);
		_network_gradient_b->calc_gradient(&loss);
		_update_rule_b->calc_update(_network_gradient_b->get_gradient(), _alpha);
		_network_b->update(_update_rule_b->get_update());
	}

	return Qs0a;
}

Tensor* DoubleQLearning::get_output(Tensor* p_state) const
{
	_network_a->activate(p_state);
	_network_b->activate(p_state);

	for(int i = 0; i < _action_count; i++)
	{
		(*_output)[i] = (_network_a->get_output()->at(i) + _network_b->get_output()->at(i)) / 2;
	}

	return _output;
}

int DoubleQLearning::calc_max_qa_index(Tensor* p_state, NeuralNetwork* p_network) const {
	p_network->activate(p_state);
	return p_network->get_output()->max_value_index();
}
