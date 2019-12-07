#include "A3C.h"
#include "RuleFactory.h"
#include "Encoder.h"
#include "RandomGenerator.h"
#include <omp.h>

using namespace Coeus;

A3C::A3C(std::vector<IEnvironment*>& p_env_array, 
		NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma, 
		NeuralNetwork* p_actor, GRADIENT_RULE p_actor_update_rule, float p_actor_alpha) :
	A2C(p_env_array, p_critic, p_critic_update_rule, p_critic_alpha, p_gamma, p_actor, p_actor_update_rule, p_actor_alpha)
{
}

A3C::~A3C()
= default;

void A3C::train(int p_t_max, int p_T_max)
{
	const size_t env_size = _env_array.size();

	omp_lock_t writelock;
	omp_init_lock(&writelock);
	
	const bool is_pooling = Tensor::is_enabled_pooling();
	if (is_pooling) Tensor::enable_pooling(false);

	int t_global = 0;
	
	#pragma omp parallel for
	for (int i = 0; i < env_size; i++)
	{
		float R;
		Tensor state0;
		Tensor state1;
		Tensor action({ _env_array[i]->ACTION_DIM() }, Tensor::ZERO);

		while(t_global < p_T_max)
		{
			_sample_buffer[i].clear();

			state0 = _env_array[i]->get_state();

			for (int roll = 0; roll < p_t_max; roll++)
			{
				if (_env_array[i]->is_finished())
				{
					break;
				}
				_actor_array[i]->activate(&state0);

				const int action0 = RandomGenerator::get_instance().choice(_actor_array[i]->get_output()->arr(), _env_array[i]->ACTION_DIM());
				Encoder::one_hot(action, action0);

				_env_array[i]->do_action(action);
				state1 = _env_array[i]->get_state();

				_sample_buffer[i].emplace_back(&state0, &action, &state1, _env_array[i]->get_reward(), _env_array[i]->is_finished());

				state0 = state1;

			}

			if (_env_array[i]->is_finished()) {
				_env_array[i]->reset();
				R = 0;
			}
			else
			{
				_critic_array[i]->activate(&state0);
				R = _critic_array[i]->get_output()->at(0);
			}

			for (int roll = _sample_buffer[i].size() - 1; roll >= 0; roll--)
			{
				R = _sample_buffer[i][roll].r + _gamma * R;

				_critic_array[i]->activate(&_sample_buffer[i][roll].s0);
				const float Vs0 = _critic_array[i]->get_output()->at(0);

				Tensor loss({ 1 }, Tensor::VALUE, Vs0 - R);
				_critic_gradient_array[i]->calc_gradient(&loss);

				const Gradient actor_d_gradient = _policy_gradient[i]->get_gradient(&_sample_buffer[i][roll].s0, _sample_buffer[i][roll].a.max_value_index(), R);
				const Gradient critic_d_gradient = _critic_gradient_array[i]->get_gradient();

				_actor_d_gradient_array[i] += actor_d_gradient;
				_critic_d_gradient_array[i] += critic_d_gradient;
			}

			omp_set_lock(&writelock);
			_critic_rule->calc_update(_critic_d_gradient_array[i]);
			_critic->update(_critic_rule->get_update());
			_actor_rule->calc_update(_actor_d_gradient_array[i]);
			_actor->update(_actor_rule->get_update());
			t_global++;
			omp_unset_lock(&writelock);

			_critic_d_gradient_array[i].fill(0);
			_actor_d_gradient_array[i].fill(0);
		}
	}

	omp_destroy_lock(&writelock);
	if (is_pooling) Tensor::enable_pooling(true);
}
