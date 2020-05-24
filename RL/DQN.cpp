#include "DQN.h"

DQN::DQN(neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma, int p_memory_size, int p_sample, int p_target_update_frequency) : Qlearning(p_critic, p_critic_optimizer, p_gamma),
	_sample_size(p_sample),
	_target_update_frequency(p_target_update_frequency),
	_target_update_step(0)
{
	_memory = new replay_buffer<mdp_transition>(p_memory_size);
	_critic_target = *p_critic;
}

DQN::~DQN()
{
	delete _memory;
}

void DQN::train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final)
{
	_memory->add_item(mdp_transition(*p_state, *p_action, *p_next_state, p_reward, p_final));

	if (_memory->size() >= _sample_size)
	{
		process_sample();

		_network->backward(critic_loss_function(&_batch_state, &_batch_action, &_batch_next_state, &_batch_reward, &_batch_mask));
		_optimizer->update();

		_target_update_step++;
		if (_target_update_step == _target_update_frequency)
		{
			_critic_target.copy_params(*_network);
			_target_update_step = 0;
		}
	}
}

tensor& DQN::critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, tensor* p_reward, tensor* p_mask)
{
	tensor& q_next_values = _critic_target.forward(p_next_state);
	vector<int> a_max_index = q_next_values.max_index(0);
	const tensor max_q_values = q_next_values.gather(a_max_index);

	vector<int> a0_index = p_action->max_index(0);
	const tensor q_values = _network->forward(p_state).gather(a0_index);

	_loss = tensor::zero_like(q_next_values);

	for (int i = 0; i < _sample_size; i++)
	{
		_loss[a0_index[i]] = (q_values[i] - ((*p_reward)[i] + _gamma * (*p_mask)[i] * max_q_values[i])) / _sample_size;
	}

	return _loss;
}

void DQN::process_sample()
{
	vector<mdp_transition*>& sample = _memory->sample(_sample_size);

	_batch_reward.resize({ _sample_size , 1 });
	_batch_mask.resize({ _sample_size , 1 });

	vector<tensor*> states;
	vector<tensor*> actions;
	vector<tensor*> next_states;

	for (int i = 0; i < sample.size(); i++)
	{
		states.push_back(&sample[i]->s0);
		actions.push_back(&sample[i]->a);
		next_states.push_back(&sample[i]->s1);
		_batch_reward[i] = sample[i]->r;
		_batch_mask[i] = sample[i]->final ? 0.f : 1.f;
	}

	tensor::concat(states, _batch_state, 1);
	tensor::concat(actions, _batch_action, 1);
	tensor::concat(next_states, _batch_next_state, 1);
}
