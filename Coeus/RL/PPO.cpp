#include "PPO.h"
#include "RuleFactory.h"
#include "RandomGenerator.h"

using namespace Coeus;


/**
 * \brief Proximal Policy Optimization algorithm constructor
 * \param p_network_critic neural network approximating value function
 * \param p_rule_critic gradient update rule for critic network
 * \param p_alpha_critic learning rate for critic network
 * \param p_gamma discount factor
 * \param p_lambda advantage estimation smoothing factor
 * \param p_network_actor neural network approximating policy
 * \param p_rule_actor gradient update rule for actor network
 * \param p_alpha_actor learning rate for actor network
 * \param p_trajectory_size length of sample trajectory collected from the environment
 */
PPO::PPO(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, const float p_gamma, const float p_lambda, NeuralNetwork* p_network_actor, GRADIENT_RULE p_rule_actor, float p_alpha_actor, const size_t p_trajectory_size) :
	_trajectory_size(p_trajectory_size)
{
	_critic = new GAE(p_network_critic, p_rule_critic, p_alpha_critic, p_gamma, p_lambda);
	_actor_old = p_network_actor;
	_actor_new = new NeuralNetwork(*p_network_actor, true);
	
	_actor_gradient = new NetworkGradient(p_network_actor);
	_actor_old_rule = RuleFactory::create_rule(p_rule_actor, _actor_old, p_alpha_actor);
	_actor_new_rule = RuleFactory::create_rule(p_rule_actor, _actor_new, p_alpha_actor);
	_policy_gradient = new PolicyGradient(_actor_new);
	_epsilon = 2e-1f;
}

PPO::~PPO()
{
	delete _critic;
	delete _actor_gradient;
	delete _actor_old_rule;
	delete _actor_new_rule;
	delete _actor_new;
	delete _policy_gradient;
}

/**
 * \brief Training method which collects samples from the environment into buffer and when the size reaches limit set int the constructor it starts training of both networks
 * \param p_state0 state in time t0
 * \param p_action action in time t0
 * \param p_state1 state in time t1
 * \param p_reward reward from transition s0 -> a -> s1
 * \param p_final state1 terminal state flag 
 */
void PPO::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final)
{
	_trajectory.emplace_back(p_state0, p_action, p_state1, p_reward, p_final);

	if (_trajectory.size() == _trajectory_size || p_final)
	{
		Gradient g_new;
		g_new.init(_actor_new);
		Gradient g_old;
		g_old.init(_actor_old);
		
		const Tensor advantages = _critic->get_advantages(_trajectory);
		_critic->train(_trajectory);

		//Tensor actor_input({static_cast<int>(_trajectory_size), _actor->get_input_dim() }, Tensor::ZERO);
		
		for(int i = 0; i < _trajectory.size(); i++)
		{
			g_new += _policy_gradient->get_gradient(&_trajectory[i].s0, _trajectory[i].a.max_value_index(), advantages[i]);
		}

		_actor_new_rule->calc_update(g_new);
		_actor_new->update(_actor_new_rule->get_update());
		

		Tensor loss({ _actor_old->get_output_dim() }, Tensor::ZERO);
		
		for (int i = 0; i < _trajectory.size(); i++)
		{
			loss.fill(0);
			_actor_old->activate(&_trajectory[i].s0);
			_actor_new->activate(&_trajectory[i].s0);
			const int action = _trajectory[i].a.max_value_index();

			const float ratio = exp(log(_actor_new->get_output()->at(action)) - log(_actor_old->get_output()->at(action)));
			float p1 = ratio * advantages[i];
			float p2 = clip(ratio, 1 - _epsilon, 1 + _epsilon) * advantages[i];
			loss[action] = -min(p1, p2);

			_actor_gradient->calc_gradient(&loss);
			g_old += _actor_gradient->get_gradient();
		}

		_actor_old_rule->calc_update(g_old);
		_actor_old->update(_actor_old_rule->get_update());
		_actor_new->copy_params(_actor_old);
		
		_trajectory.clear();
	}
}

/**
 * \brief Selects action according to stochastic policy for the input state 
 * \param p_state input state
 * \return one-hot encoded action in tensor
 */
Tensor PPO::get_action(Tensor* p_state) const
{
	Tensor result({ _actor_old->get_output_dim() }, Tensor::ZERO);
	_actor_old->activate(p_state);
	const int action = RandomGenerator::get_instance().choice(_actor_old->get_output()->arr(), _actor_old->get_output_dim());
	result[action] = 1;

	return result;
}

float PPO::clip(const float p_value, const float p_lower_bound, const float p_upper_bound) const
{
	float result = p_value;
	result = max(result, p_lower_bound);
	result = min(result, p_upper_bound);

	return result;
}
