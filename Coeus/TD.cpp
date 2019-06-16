#include "TD.h"
#include "RuleFactory.h"

using namespace Coeus;

TD::TD(NeuralNetwork* p_network, GradientAlgorithm* p_optimizer, const float p_gamma, const float p_alpha): _alpha(p_alpha), _gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = nullptr;
	_update_rule = nullptr;
	_optimizer = p_optimizer;
}

TD::TD(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda) :
	_gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new GeneralTDRule(_network_gradient, RuleFactory::create_rule(p_grad_rule, _network_gradient, p_alpha), p_alpha, p_gamma, p_lambda);
	_optimizer = nullptr;
}

TD::~TD()
{
	delete _network_gradient;
	delete _update_rule;
}

float TD::train(Tensor* p_state0, Tensor* p_state1, const float p_reward) const
{
	_network->activate(p_state0);
	const float Vs0 = _network->get_output()->at(0);
	_network->activate(p_state1);
	const float Vs1 = _network->get_output()->at(0);

	const float delta = p_reward + _gamma * Vs1 - Vs0;

	if (_update_rule != nullptr)
	{
		_network_gradient->calc_gradient();
		_update_rule->calc_update(_network_gradient->get_gradient(), delta, 0);
		_network->update(_update_rule->get_update());
	}

	if (_optimizer != nullptr)
	{
		Tensor target = *_network->get_output();
		target[0] += _alpha * delta;

		_optimizer->train(p_state0, &target);
	}

	return delta;
}
