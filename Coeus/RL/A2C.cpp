#include "A2C.h"
#include "RuleFactory.h"
#include "RandomGenerator.h"
#include "Encoder.h"

using namespace Coeus;

A2C::A2C(vector<IEnvironment*> &p_env_array,
		NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma, float p_lambda, 
		NeuralNetwork* p_actor,	GRADIENT_RULE p_actor_update_rule, float p_actor_alpha):
	_env_array(p_env_array)
{
	const size_t env_size = p_env_array.size();
	_critic = p_critic;
	_critic_array = new NeuralNetwork*[env_size];
	_critic_d_gradient.init(_critic);
	_critic_d_gradient_array = new Gradient[env_size];
	_advantage_estimation = new GAE*[env_size];
	_critic_rule = RuleFactory::create_rule(p_critic_update_rule, p_critic, p_critic_alpha);

	_actor = p_actor;
	_actor_array = new NeuralNetwork*[env_size];
	_actor_d_gradient.init(_actor);
	_actor_d_gradient_array = new Gradient[env_size];
	_policy_gradient = new PolicyGradient*[env_size];
	_actor_rule = RuleFactory::create_rule(p_actor_update_rule, p_actor, p_actor_alpha);

	_sample_buffer = new vector<DQItem>[env_size];

	for (int i = 0; i < env_size; i++)
	{
		_critic_array[i] = new NeuralNetwork(*_critic);
		_critic_d_gradient_array[i].init(_critic);
		_advantage_estimation[i] = new GAE(_critic_array[i], p_gamma, p_lambda);

		_actor_array[i] = new NeuralNetwork(*_actor);
		_actor_d_gradient_array[i].init(_actor);
		_policy_gradient[i] = new PolicyGradient(_actor_array[i]);
	}
}

A2C::~A2C()
{
	const size_t env_size = _env_array.size();
	
	for (int i = 0; i < env_size; i++)
	{
		delete _critic_array[i];
		delete _advantage_estimation[i];

		delete _actor_array[i];
		delete _policy_gradient[i];
	}

	delete[] _critic_array;
	delete[] _advantage_estimation;	

	delete[] _actor_array;
	delete[] _policy_gradient;

	delete[] _sample_buffer;
}

void A2C::train(int p_rollout_size)
{
	const size_t env_size = _env_array.size();

	const bool is_pooling = Tensor::is_enabled_pooling();
	
	if (is_pooling) Tensor::enable_pooling(false);
	
	#pragma omp parallel for
	for(int i = 0; i < env_size; i++)
	{
		Tensor state0;
		Tensor state1;
		Tensor action({ _env_array[i]->ACTION_DIM() }, Tensor::ZERO);
		_sample_buffer[i].clear();

		state0 = _env_array[i]->get_state();
		
		for(int roll = 0; roll < p_rollout_size; roll++)
		{
			if (_env_array[i]->is_finished())
			{
				_env_array[i]->reset();
			}
			_actor_array[i]->activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(_actor_array[i]->get_output()->arr(), _env_array[i]->ACTION_DIM());
			Encoder::one_hot(action, action0);

			_env_array[i]->do_action(action);
			state1 = _env_array[i]->get_state();

			_sample_buffer[i].emplace_back(&state0, &action, &state1, _env_array[i]->get_reward(), _env_array[i]->is_finished());
			
			state0 = state1;
		}

		_advantage_estimation[i]->set_sample(_sample_buffer[i]);
		vector<float> advantages = _advantage_estimation[i]->get_advantages();		

		for (int roll = 0; roll < p_rollout_size; roll++)
		{
			const Gradient actor_d_gradient = _policy_gradient[i]->get_gradient(&_sample_buffer[i][roll].s0, _sample_buffer[i][roll].a.max_value_index(), advantages[roll]);
			const Gradient critic_d_gradient = _advantage_estimation[i]->get_gradient(&_sample_buffer[i][roll].s0, advantages[roll]);

			_actor_d_gradient_array[i] += actor_d_gradient;
			_critic_d_gradient_array[i] += critic_d_gradient;
		}
	}

	for (int i = 0; i < env_size; i++)
	{
		_critic_d_gradient += _critic_d_gradient_array[i];
		_critic_d_gradient_array[i].fill(0);

		_actor_d_gradient += _actor_d_gradient_array[i];
		_actor_d_gradient_array[i].fill(0);
	}

	_critic_rule->calc_update(_critic_d_gradient);
	_critic->update(_critic_rule->get_update());
	_actor_rule->calc_update(_actor_d_gradient);
	_actor->update(_actor_rule->get_update());

	_critic_d_gradient.fill(0);
	_actor_d_gradient.fill(0);

	if (is_pooling) Tensor::enable_pooling(true);
}
