#include "DDPG.h"
#include <iostream>
#include "CLAB.h"


DDPG::DDPG(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma, const int p_memory_size, const int p_sample_size, const float p_tau) :
	_actor(p_actor),
	_critic(p_critic),
	_gamma(p_gamma), 
	_tau(p_tau),
	_actor_optimizer(p_actor_optimizer),
	_critic_optimizer(p_critic_optimizer),
	_sample_size(p_sample_size),
	_forward_model(nullptr),
	_metacritic(nullptr)
{
	_memory = new replay_buffer<mdp_transition>(p_memory_size);
	_actor_target = *p_actor;
	_critic_target = *p_critic;
}

DDPG::~DDPG()
{
	delete _memory;
}

tensor& DDPG::get_action(tensor* p_state) const
{
	return _actor->forward(p_state);
}

void DDPG::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_memory->add_item(mdp_transition(*p_state, *p_action, *p_next_state, p_reward, p_final));

	if (_memory->size() >= _sample_size)
	{
		process_sample();

		_critic->backward(critic_loss_function());
		_critic_optimizer->update();

		_actor->backward(actor_loss_function());
		_actor_optimizer->update();

		/*
		if (_forward_model != nullptr)
		{
			_forward_model->train(&batch_state, &batch_action, &batch_next_state);
		}

		if (_metacritic != nullptr)
		{
			_metacritic->train(&batch_state, &batch_action, &batch_next_state);
		}
		*/

		_actor_target.copy_params(*_actor, _tau);
		_critic_target.copy_params(*_critic, _tau);
	}
}

void DDPG::add_motivation(forward_model* p_motivation)
{
	_forward_model = p_motivation;
}

void DDPG::add_motivation(metacritic* p_motivation)
{
	_metacritic = p_motivation;
}

void DDPG::process_sample()
{
	vector<mdp_transition*>& sample = _memory->sample(_sample_size);
	
	batch_reward.resize({ _sample_size , 1 });
	batch_mask.resize({ _sample_size , 1 });

	vector<tensor*> states;
	vector<tensor*> actions;
	vector<tensor*> next_states;

	for(int i = 0; i < sample.size(); i++)
	{
		states.push_back(&sample[i]->s0);
		actions.push_back(&sample[i]->a);
		next_states.push_back(&sample[i]->s1);
		batch_reward[i] = sample[i]->r;
		batch_mask[i] = sample[i]->final ? 0.f : 1.f;
	}

	tensor::concat(states, batch_state, 1);
	tensor::concat(actions, batch_action, 1);
	tensor::concat(next_states, batch_next_state, 1);

	if (_forward_model != nullptr)
	{
		tensor& internal_reward = _forward_model->reward(&batch_state, &batch_action, &batch_next_state);
		batch_reward += internal_reward;
	}

	if (_metacritic != nullptr)
	{
		tensor& internal_reward = _metacritic->reward(&batch_state, &batch_action, &batch_next_state);
		batch_reward += internal_reward;
	}
}

tensor& DDPG::actor_loss_function()
{
	map<string, tensor*> critic_input;
	critic_input["hidden0"] = &batch_state;
	critic_input["hidden1"] = &_actor->forward(&batch_state);
	_critic->forward(critic_input);

	tensor delta = tensor::value({ _sample_size, 1 }, -1.f / _sample_size);
	_actor_loss = *_critic->backward(delta)["hidden1"];

	return _actor_loss;
}

tensor& DDPG::critic_loss_function()
{	
	map<string, tensor*> critic_input;
	critic_input["hidden0"] = &batch_state;
	critic_input["hidden1"] = &batch_action;

	tensor& q_value = _critic->forward(critic_input);

	map<string, tensor*> critic_target_input;
	critic_target_input["hidden0"] = &batch_next_state;
	critic_target_input["hidden1"] = &_actor_target.forward(&batch_next_state);

	tensor& max_q_value = _critic_target.forward(critic_target_input);
	_critic_loss = tensor::zero_like(q_value);

	const int size = _sample_size / segment;

	float* lx = _critic_loss.data();
	float* qx = q_value.data();
	float* maxqx = max_q_value.data();
	float* mx = batch_mask.data();
	float* rx = batch_reward.data();

	const __m256 gamma256 = _mm256_broadcast_ss(&_gamma);
	float sample_size = _sample_size;
	const __m256 sample_size256 = _mm256_broadcast_ss(&sample_size);

	for (int i = 0; i < size; i++)
	{
		const __m256 mx256 = _mm256_load_ps(mx);
		const __m256 rx256 = _mm256_load_ps(rx);
		const __m256 qx256 = _mm256_load_ps(qx);
		const __m256 maxqx256 = _mm256_load_ps(maxqx);
		const __m256 lx256 = _mm256_div_ps(_mm256_sub_ps(qx256, _mm256_add_ps(rx256, _mm256_mul_ps(gamma256, _mm256_mul_ps(mx256, maxqx256)))), sample_size256);
		_mm256_storeu_ps(lx, lx256);

		mx += segment;
		rx += segment;
		qx += segment;
		maxqx += segment;
		lx += segment;		
	}

	for (int i = size * segment; i < _sample_size; i++)
	{
		*lx++ = (*qx++ - (*rx++ + _gamma * *mx++ * *maxqx++)) / _sample_size;
	}

	return _critic_loss;
}
