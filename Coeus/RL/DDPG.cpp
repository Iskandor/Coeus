#include "DDPG.h"
#include "RuleFactory.h"
#include "QuadraticCost.h"
#include "TensorOperator.h"

using namespace Coeus;


DDPG::DDPG(	NeuralNetwork* p_network_critic, GRADIENT_RULE p_critic_rule, float p_critic_alpha, float p_gamma,
			NeuralNetwork* p_network_actor, GRADIENT_RULE p_actor_rule, float p_actor_alpha, 
			int p_buffer_size, int p_sample_size):
	_network_actor(p_network_actor), 
	_network_critic(p_network_critic),
	_gamma(p_gamma), 
	_sample_size(p_sample_size) 
{
	_network_actor_gradient = new NetworkGradient(p_network_actor);
	_network_critic_gradient = new NetworkGradient(p_network_critic);

	_network_actor_target = new NeuralNetwork(*p_network_actor, true);
	_network_critic_target = new NeuralNetwork(*p_network_critic, true);

	_update_rule_actor = RuleFactory::create_rule(p_actor_rule, _network_actor, p_actor_alpha);
	_update_rule_critic = RuleFactory::create_rule(p_critic_rule, _network_critic, p_critic_alpha);

	_buffer = new ReplayBuffer<DCQItem>(p_buffer_size);

	_target = new Tensor({ p_sample_size, p_network_critic->get_output_dim() }, Tensor::ZERO);
	_actor_input = new Tensor({ p_sample_size, p_network_actor->get_input_dim() }, Tensor::ZERO);
	_critic_input = new Tensor({ p_sample_size, p_network_critic->get_input_dim() }, Tensor::ZERO);
	_critic_input2 = new Tensor({ p_sample_size, p_network_critic->get_input_dim() }, Tensor::ZERO);
}

DDPG::~DDPG()
{
	delete _network_actor_gradient;
	delete _network_critic_gradient;
	delete _buffer;
	delete _actor_input;
	delete _target;
	delete _critic_input;
	delete _critic_input2;
}

float DDPG::train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, const float p_reward, bool p_final) const
{
	_buffer->add_item(new DCQItem(p_state0, p_action0, p_state1, p_reward, p_final));

	float error = 0;

	if (_buffer->get_size() >= _sample_size) {
		vector<DCQItem*>* sample = _buffer->get_sample(_sample_size);

		_actor_input->reset_index();
		_critic_input->reset_index();
		_critic_input2->reset_index();
		_target->reset_index();

		for (auto& s : *sample)
		{
			_actor_input->push_back(&s->s0);
			_critic_input->push_back(&s->s0);
			_critic_input->push_back(&s->a);
			_network_actor->activate(&s->s0);
			_critic_input2->push_back(&s->s0);
			_critic_input2->push_back(_network_actor->get_output());
		}

		_network_critic->activate(_critic_input);

		for (size_t i = 0; i < sample->size(); i++)
		{
			DCQItem* s = (*sample)[i];
			if (s->final) {
				_target->push_back(s->r);
			}
			else {
				const float maxQs1a = calc_max_qa(&s->s1);
				_target->push_back(s->r + _gamma * maxQs1a - _network_critic->get_output()->at(i));
			}
		}
		
		Tensor critic_loss = *_network_critic->get_output() - *_target;
		
		_network_critic_gradient->calc_gradient(&critic_loss);
		_update_rule_critic->calc_update(_network_critic_gradient->get_gradient());

		_network_critic->activate(_critic_input2);
		_network_critic_gradient->calc_gradient();
		_network_actor->activate(_actor_input);
		
		Tensor actor_loss = -_network_critic_gradient->get_input_gradient(_sample_size, _network_critic->get_input_dim() - _network_actor->get_output_dim(), _network_actor->get_output_dim());
		
		_network_actor_gradient->calc_gradient(&actor_loss);
		_update_rule_actor->calc_update(_network_actor_gradient->get_gradient());

		_network_critic->update(_update_rule_critic->get_update());
		_network_actor->update(_update_rule_actor->get_update());

		_network_critic_target->polyak_averaging(0.99, _network_critic);
		_network_actor_target->polyak_averaging(0.99, _network_actor);
	}


	return error;
}

Tensor DDPG::get_action(Tensor* p_state, const float p_sigma)
{
	Tensor output({ _network_actor->get_output_dim() }, Tensor::ZERO);
	_network_actor->activate(p_state);

	for (int i = 0; i < _network_actor->get_output_dim(); i++)
	{
		const float rand = p_sigma > 0 ? RandomGenerator::get_instance().normal_random(0, p_sigma) : 0;
		output[i] = _network_actor->get_output()->at(i) + rand;
	}

	return output;
}

float DDPG::calc_max_qa(Tensor* p_state) const {
	_network_actor_target->activate(p_state);

	Tensor i({ _network_critic_target->get_input_dim() }, Tensor::ZERO);
	i.push_back(p_state);
	i.push_back(_network_actor_target->get_output());

	_network_critic_target->activate(&i);

	return _network_critic_target->get_output()->at(0);
}
