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
	const int env_size = p_env_array.size();
	_critic = p_critic;
	_critic_array = new NeuralNetwork*[env_size];
	_critic_param_array = new ParamModel*[env_size];
	_advantage_estimation = new GAE*[env_size];

	_actor = p_actor;
	_actor_array = new NeuralNetwork*[env_size];
	_actor_param_array = new ParamModel*[env_size];
	_policy_gradient = new Actor*[env_size];

	_sample_buffer = new vector<DQItem>[env_size];

	for (int i = 0; i < env_size; i++)
	{
		_critic_array[i] = _critic->clone();
		_critic_param_array[i] = _critic_array[i];
		_advantage_estimation[i] = new GAE(_critic_array[i], p_critic_update_rule, p_critic_alpha, p_gamma, p_lambda);

		_actor_array[i] = _actor->clone();
		_actor_param_array[i] = _actor_array[i];
		_policy_gradient[i] = new Actor(_actor_array[i], p_actor_update_rule, p_actor_alpha);
	}
}

A2C::~A2C()
{
	const int env_size = _env_array.size();
	
	for (int i = 0; i < env_size; i++)
	{
		delete _critic_array[i];
		delete _advantage_estimation[i];

		delete _actor_array[i];
		delete _policy_gradient[i];
	}

	delete[] _critic_array;
	delete[] _critic_param_array;
	delete[] _advantage_estimation;	

	delete[] _actor_array;
	delete[] _actor_param_array;
	delete[] _policy_gradient;

	delete[] _sample_buffer;
}

void A2C::train(int p_rollout_size) const
{
	const int env_size = _env_array.size();

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

		_critic_array[i]->copy_params(_critic);
		_actor_array[i]->copy_params(_actor);
		
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
		_advantage_estimation[i]->train();

		for (int roll = 0; roll < p_rollout_size; roll++)
		{
			_policy_gradient[i]->train(&_sample_buffer[i][roll].s0, _sample_buffer[i][roll].a.max_value_index(), advantages[roll]);
		}
	}

	_critic->average_params(_critic_param_array, env_size);
	_actor->average_params(_actor_param_array, env_size);

	if (is_pooling) Tensor::enable_pooling(true);
}
