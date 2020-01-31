#include "CACER.h"

using namespace Coeus;

CACER::CACER(NeuralNetwork* p_critic, GRADIENT_RULE p_critic_rule, float p_critic_alpha, float p_gamma,
			 NeuralNetwork* p_actor, GRADIENT_RULE p_actor_rule, float p_actor_alpha, int p_buffer_size, int p_sample_size) :
	CACLA(p_critic, p_critic_rule, p_critic_alpha, p_gamma, p_actor, p_actor_rule, p_actor_alpha)
{
	_buffer = new ReplayBuffer<DQItem>(p_buffer_size);
	_sample_size = p_sample_size;
}

CACER::~CACER()
{
	delete _buffer;
}

void CACER::train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, float p_reward, bool p_final)
{
	_buffer->add_item(new DQItem(p_state0, p_action0, p_state1, p_reward, p_final));

	if (_buffer->get_size() >= _sample_size) {
		vector<DQItem*>* sample = _buffer->get_sample(_sample_size);

		const Tensor delta = _critic->train(sample);
		int size = 0;
		for (size_t i = 0; i < sample->size(); i++)
		{
			if (delta[i] > 0)
			{
				size++;
			}
		}

		//TODO: memory leak somewhere here!!!
		Tensor state0({ size, _actor->get_input_dim() }, Tensor::ZERO);
		Tensor action({ size, _actor->get_output_dim() }, Tensor::ZERO);

		for (size_t i = 0; i < sample->size(); i++)
		{
			if (delta[i] > 0)
			{
				state0.insert_row(&sample->at(i)->s0);
				action.insert_row(&sample->at(i)->a);
			}
		}
		
		_actor->activate(&state0);
		Tensor loss = _mse.cost_deriv(_actor->get_output(), &action);
		
		_actor_gradient->calc_gradient(&loss);
		_update_rule->calc_update(_actor_gradient->get_gradient());
		_actor->update(_update_rule->get_update());
	}
}
