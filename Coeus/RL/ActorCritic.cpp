#include "ActorCritic.h"
#include "RuleFactory.h"

using namespace Coeus;

ActorCritic::ActorCritic(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, 
						NeuralNetwork* p_network_actor, GRADIENT_RULE p_rule_actor, float p_alpha_actor) :
	_network_critic(p_network_critic),
	_network_actor(p_network_actor)
{
	_critic = new TD(p_network_critic, p_rule_critic, p_alpha_critic, p_gamma);
	_actor = new PolicyGradient(p_network_actor);

	_rule_actor = RuleFactory::create_rule(p_rule_actor, _network_actor, p_alpha_actor);
}

ActorCritic::~ActorCritic()
{
	delete _critic;
	delete _actor;
}

void ActorCritic::train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, float p_reward, bool p_final) const
{
	const float delta = _critic->train(p_state0, p_state1, p_reward, p_final);
	_rule_actor->calc_update(_actor->get_gradient(p_state0, p_action0->max_value_index(), delta));
	_network_actor->update(_rule_actor->get_update());
}


