#include "PPO.h"
#include "RuleFactory.h"

using namespace Coeus;


PPO::PPO(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, const float p_gamma, const float p_lambda, NeuralNetwork* p_network_actor, GRADIENT_RULE p_rule_actor, float p_alpha_actor, const size_t p_trajectory_size) :
	_trajectory_size(p_trajectory_size)
{
	_critic = new GAE(p_network_critic, p_rule_critic, p_alpha_critic, p_gamma, p_lambda);
	_actor = p_network_actor;
	_actor_gradient = new NetworkGradient(p_network_actor);
	_actor_rule = RuleFactory::create_rule(p_rule_actor, p_network_actor, p_alpha_actor);
}

PPO::~PPO()
{
	delete _critic;
	delete _actor_gradient;
	delete _actor_rule;
}

void PPO::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final)
{
	_trajectory.emplace_back(p_state0, p_action, p_state1, p_reward, p_final);

	if (_trajectory.size() == _trajectory_size)
	{
		Tensor advantages = _critic->get_advantages(_trajectory);
		_critic->train(_trajectory);
		
		for(auto sample : _trajectory)
		{
			
		}
		
		_trajectory.clear();
	}
}
