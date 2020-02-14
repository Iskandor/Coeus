#include "DDPG.h"
#include "RuleFactory.h"

using namespace Coeus;

/**
 * \brief DDPG algorithm constructor
 * \param p_network_critic critic neural network
 * \param p_critic_rule update rule for critic training
 * \param p_critic_alpha critic learning rate
 * \param p_gamma critic discount factor
 * \param p_network_actor actor neural network
 * \param p_actor_rule update rule for actor training
 * \param p_actor_alpha actor learning rate
 * \param p_buffer_size size of replay buffer
 * \param p_sample_size size of sample taken from the replay buffer
 */
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

	_buffer = new ReplayBuffer<DQItem>(p_buffer_size);

	_target = Tensor({ p_sample_size, p_network_critic->get_output_dim() }, Tensor::ZERO);
	_batch_input_s0 = Tensor({ p_sample_size, p_network_actor->get_input_dim() }, Tensor::ZERO);
	_batch_input_s1 = Tensor({ p_sample_size, p_network_actor->get_input_dim() }, Tensor::ZERO);
	_critic_input_a0 = Tensor({ p_sample_size, p_network_critic->get_input_dim() }, Tensor::ZERO);
	_critic_input_a = Tensor({ p_sample_size, p_network_critic->get_input_dim() }, Tensor::ZERO);
}

DDPG::~DDPG()
{
	delete _network_actor_gradient;
	delete _network_critic_gradient;
	delete _update_rule_actor;
	delete _update_rule_critic;
	delete _buffer;
}

/**
 * \brief Training method updating actor and critic on minibatch taken from the replay buffer
 * \param p_state0 agent's state from time t0
 * \param p_action0 action taken in time t0
 * \param p_state1 agent's state from time t1 (successor state)
 * \param p_reward reward taken from transition state0 - action0 - state1
 * \param p_final flag indicating that state1 is terminal
 */
void DDPG::train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, const float p_reward, bool p_final)
{
	_buffer->add_item(new DQItem(p_state0, p_action0, p_state1, p_reward, p_final));

	if (_buffer->get_size() >= _sample_size) {
		Tensor actor_output_row({ _network_actor->get_output_dim() }, Tensor::ZERO);
		vector<DQItem*>* sample = _buffer->get_sample(_sample_size);

		_batch_input_s0.reset_index();
		_batch_input_s1.reset_index();
		_critic_input_a0.reset_index();
		_critic_input_a.reset_index();

		for (auto& s : *sample)
		{
			_batch_input_s0.insert_row(&s->s0);
			_batch_input_s1.insert_row(&s->s1);
			_critic_input_a0.push_back(&s->s0);
			_critic_input_a0.push_back(&s->a);
		}
		
		_network_actor->activate(&_batch_input_s0);
		_network_critic->activate(&_critic_input_a0);

		_critic_input_a.push_back(&_batch_input_s0);
		_critic_input_a.push_back(_network_actor->get_output());
		
		const Tensor* maxQs1a = calc_max_qa();
		
		for (size_t i = 0; i < sample->size(); i++)
		{
			DQItem* s = (*sample)[i];

			if (s->final) {
				_target[i] = s->r;
			}
			else {
				
				_target[i] = s->r + _gamma * (*maxQs1a)[i] - _network_critic->get_output()->at(i);
			}
		}
		
		Tensor critic_loss = *_network_critic->get_output() - _target;
		
		_network_critic_gradient->calc_gradient(&critic_loss);
		_update_rule_critic->calc_update(_network_critic_gradient->get_gradient());

		_network_critic->activate(&_critic_input_a);
		critic_loss.fill(-1);
		_network_critic_gradient->calc_gradient(&critic_loss);
		_network_actor->activate(&_batch_input_s0);
		
		Tensor actor_loss = _network_critic_gradient->get_input_gradient(_sample_size, _network_critic->get_input_dim() - _network_actor->get_output_dim(), _network_actor->get_output_dim());
		
		_network_actor_gradient->calc_gradient(&actor_loss);
		_update_rule_actor->calc_update(_network_actor_gradient->get_gradient());

		_network_critic->update(_update_rule_critic->get_update());
		_network_actor->update(_update_rule_actor->get_update());

		_network_critic_target->polyak_averaging(0.999, _network_critic);
		_network_actor_target->polyak_averaging(0.999, _network_actor);
	}
}

/**
 * \brief Return action as output from the actor network
 * \param p_state actual state where the agent is choosing its next action
 * \return 
 */
Tensor* DDPG::get_action(Tensor* p_state) const
{
	_network_actor->activate(p_state);
	return _network_actor->get_output();
}

Tensor* DDPG::calc_max_qa() {
	_network_actor_target->activate(&_batch_input_s1);

	Tensor i({ _sample_size, _network_critic_target->get_input_dim() }, Tensor::ZERO);
	i.push_back(&_batch_input_s1);
	i.push_back(_network_actor_target->get_output());

	_network_critic_target->activate(&i);

	return _network_critic_target->get_output();
}
