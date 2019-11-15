#include "A3C.h"
#include "RuleFactory.h"
#include "Encoder.h"
#include "RandomGenerator.h"
#include <omp.h>

using namespace Coeus;

A3C::A3C(std::vector<IEnvironment*>& p_env_array, int p_t_max, 
		NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma, float p_lambda, 
		NeuralNetwork* p_actor, GRADIENT_RULE p_actor_update_rule, float p_actor_alpha) :
	_env_array(p_env_array),
	_t_max(p_t_max)
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

	_t = new int[env_size];
	
	for (int i = 0; i < env_size; i++)
	{
		_critic_array[i] = _critic->clone();
		_critic_d_gradient_array[i].init(_critic);
		_advantage_estimation[i] = new GAE(_critic_array[i], p_gamma, p_lambda);

		_actor_array[i] = _actor->clone();
		_actor_d_gradient_array[i].init(_actor);
		_policy_gradient[i] = new PolicyGradient(_actor_array[i]);
		
		_t[i] = 0;
	}
}

A3C::~A3C()
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
	delete[] _t;
}

void A3C::train(int p_rollout_size)
{
	const size_t env_size = _env_array.size();

	omp_lock_t writelock;
	omp_init_lock(&writelock);
	
	const bool is_pooling = Tensor::is_enabled_pooling();
	if (is_pooling) Tensor::enable_pooling(false);

	for (int roll = 0; roll < p_rollout_size; roll++)
	{
		#pragma omp parallel for
		for (int i = 0; i < env_size; i++)
		{
			Tensor state0;
			Tensor state1;
			Tensor action({ _env_array[i]->ACTION_DIM() }, Tensor::ZERO);
			_sample_buffer[i].clear();

			state0 = _env_array[i]->get_state();

			_actor_array[i]->activate(&state0);

			const int action0 = RandomGenerator::get_instance().choice(_actor_array[i]->get_output()->arr(), _env_array[i]->ACTION_DIM());
			Encoder::one_hot(action, action0);

			_env_array[i]->do_action(action);
			state1 = _env_array[i]->get_state();

			_sample_buffer[i].emplace_back(&state0, &action, &state1, _env_array[i]->get_reward(), _env_array[i]->is_finished());

			state0 = state1;

			_t[i]++;

			if (_env_array[i]->is_finished() || _t[i] == _t_max)
			{
				_env_array[i]->reset();
				_t[i] = 0;

				_advantage_estimation[i]->set_sample(_sample_buffer[i]);
				vector<float> advantages = _advantage_estimation[i]->get_advantages();
				
				for(int s = 0; s < _sample_buffer[i].size(); s++)
				{
					const Gradient actor_d_gradient(_policy_gradient[i]->get_gradient(&_sample_buffer[i][s].s0, _sample_buffer[i][s].a.max_value_index(), advantages[s]));
					const Gradient critic_d_gradient(_advantage_estimation[i]->get_gradient(&_sample_buffer[i][s].s0, advantages[s]));

					_actor_d_gradient_array[i] += actor_d_gradient;
					_critic_d_gradient_array[i] += critic_d_gradient;
				}

				omp_set_lock(&writelock);
				_critic_rule->calc_update(_critic_d_gradient_array[i]);
				_critic->update(_critic_rule->get_update());
				_actor_rule->calc_update(_actor_d_gradient_array[i]);
				_actor->update(_actor_rule->get_update());
				omp_unset_lock(&writelock);

				_critic_d_gradient_array[i].fill(0);
				_actor_d_gradient_array[i].fill(0);

				_sample_buffer->clear();
			}
		}
	}

	omp_destroy_lock(&writelock);
	if (is_pooling) Tensor::enable_pooling(true);
}
