#include "QLearning.h"
#include "NetworkGradient.h"
#include "GeneralTDRule.h"
#include "RuleFactory.h"

using namespace Coeus;

QLearning::QLearning(NeuralNetwork* p_network, GradientAlgorithm* p_optimizer, double p_gamma, double p_alpha):
	_alpha(p_alpha), _gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = nullptr;
	_update_rule = nullptr;
	_optimizer = p_optimizer;

}

QLearning::QLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, const double p_alpha, const double p_gamma, const double p_lambda):
	_alpha(0), _gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new GeneralTDRule(_network_gradient, RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha), p_alpha, p_gamma, p_lambda);
	_optimizer = nullptr;
}


QLearning::~QLearning()
{
	delete _network_gradient;
	delete _update_rule;
}

double QLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const double p_reward) const
{
	const double maxQs1a = calc_max_qa(p_state1);

	_network->activate(p_state0);
	const double Qs0a = _network->get_output()->at(p_action0);
	const double delta = p_reward + _gamma * maxQs1a - Qs0a;

	if (_update_rule != nullptr)
	{
		Tensor mask = Tensor::Zero({ _network->get_output()->size() });
		mask[p_action0] = 1;

		_network_gradient->calc_gradient(&mask);
		_update_rule->calc_update(_network_gradient->get_gradient(), delta);
		_network->update(_update_rule->get_update());
	}
	if (_optimizer != nullptr)
	{
		Tensor target = *_network->get_output();
		target[p_action0] += _alpha * delta;

		_optimizer->train(p_state0, &target);
	}

	return delta;
}

void QLearning::reset_traces() const
{
	_update_rule->reset_traces();
}

double QLearning::calc_max_qa(Tensor* p_state) const
{
	_network->activate(p_state);
	const int maxQa = _network->get_output()->max_value_index();

	return _network->get_output()->at(maxQa);
}