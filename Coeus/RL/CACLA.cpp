#include "CACLA.h"
#include "RuleFactory.h"
#include "RandomGenerator.h"
#include "QuadraticCost.h"

using namespace Coeus;

CACLA::CACLA(NeuralNetwork* p_critic, GRADIENT_RULE p_critic_rule, float p_critic_alpha, float p_gamma, NeuralNetwork* p_actor, GRADIENT_RULE p_actor_rule, float p_actor_alpha, float p_beta) :
	_actor(p_actor),
	_actor_alpha(p_actor_alpha),
	_beta(p_beta),
	_var(1)
{
	_actor_gradient = new NetworkGradient(p_actor);
	_update_rule = RuleFactory::create_rule(p_actor_rule, p_actor, p_actor_alpha);

	_critic = new TD(p_critic, p_critic_rule, p_critic_alpha, p_gamma);
}

CACLA::~CACLA()
{
	delete _critic;
	delete _update_rule;
	delete _actor_gradient;
}

void CACLA::train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, const float p_reward, const bool p_final)
{
	const float delta = _critic->train(p_state0, p_action0, p_state1, p_reward, p_final);
	
	if (_beta > 0) _var = (1 - _beta) * _var + _beta * delta * delta;

	if (delta > 0)
	{
		const float v = ceil(delta / sqrt(_var));
		_actor->activate(p_state0);

		Tensor loss = _mse.cost_deriv(_actor->get_output(), p_action0);

		_actor_gradient->calc_gradient(&loss);
		_update_rule->calc_update(_actor_gradient->get_gradient(), _actor_alpha * v);
		_actor->update(_update_rule->get_update());
	}
}

Tensor CACLA::get_action(Tensor* p_state, const float p_sigma) const
{
	Tensor output({ _actor->get_output_dim() }, Tensor::ZERO);
	_actor->activate(p_state);

	for(int i = 0; i < _actor->get_output_dim(); i++)
	{
		const float rand = p_sigma > 0.f ? RandomGenerator::get_instance().normal_random(0, p_sigma) : 0.f;
		output[i] = _actor->get_output()->at(i) + rand;
	}

	return output;
}
