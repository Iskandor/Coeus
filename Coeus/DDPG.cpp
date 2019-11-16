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

	_network_actor_target = p_network_actor->clone();
	_network_critic_target = p_network_critic->clone();

	_update_rule_actor = RuleFactory::create_rule(p_actor_rule, _network_actor, p_actor_alpha);
	_update_rule_critic = RuleFactory::create_rule(p_critic_rule, _network_critic, p_critic_alpha);

	_buffer = new ReplayBuffer<DQItem>(p_buffer_size);

	_target = Tensor::Zero({ p_sample_size, _network_critic->get_output_dim() });
	_input_critic_s0 = Tensor::Zero({ p_sample_size, _network_critic->get_input_dim() });
	_input_critic_s1 = Tensor::Zero({ p_sample_size, _network_critic->get_input_dim() });
	_input_actor_s0 = Tensor::Zero({ p_sample_size, _network_actor->get_input_dim() });
	_input_actor_s1 = Tensor::Zero({ p_sample_size, _network_actor->get_input_dim() });
	_q_target = Tensor::Zero({ p_sample_size });

	_ou_state = new Tensor({p_network_actor->get_output_dim()}, Tensor::ZERO);

	_mu = 0.0f;
	_theta = 0.15f;
	_min_sigma = 0.3f;
	_max_sigma = 0.3f;
	_decay_period = 1e5;
	_sigma = _max_sigma;
}

DDPG::~DDPG()
{
	delete _network_actor_gradient;
	delete _network_critic_gradient;
	delete _buffer;
	delete _ou_state;
}

float DDPG::train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, const float p_reward, bool p_final)
{
	_buffer->add_item(new DQItem(p_state0, p_action0, p_state1, p_reward, p_final));

	float error = 0;

	if (_buffer->get_size() >= _sample_size) {
		vector<DQItem*>* sample = _buffer->get_sample(_sample_size);

		_input_critic_s0.reset_index();
		_input_critic_s1.reset_index();
		_input_actor_s0.reset_index();
		_input_actor_s1.reset_index();
		for (auto& s : *sample)
		{
			_input_critic_s0.push_back(&s->s0);
			_input_critic_s0.push_back(&s->a);
			_input_actor_s0.push_back(&s->s0);
			_input_actor_s1.push_back(&s->s1);
		}

		_network_actor_target->activate(&_input_actor_s1);
		Tensor* actor_target_output = _network_actor_target->get_output();

		for (size_t i = 0; i < sample->size(); i++)
		{
			_input_critic_s1.push_back(&sample->at(i)->s1);
			for(int j = 0; j < _network_actor_target->get_output_dim(); j++)
			{
				_input_critic_s1.push_back(actor_target_output->at(i,j));
			}			
		}
		_network_critic_target->activate(&_input_critic_s1);
		Tensor* critic_target_output = _network_critic_target->get_output();

		for (size_t i = 0; i < sample->size(); i++) {
			_target.set(i, sample->at(i)->r + _gamma * critic_target_output->at(i));
		}

		QuadraticCost mse;
		_network_critic->activate(&_input_critic_s0);
		Tensor critic_loss = mse.cost_deriv(_network_critic->get_output(), &_target);
		_network_critic_gradient->calc_gradient(&critic_loss);
		_update_rule_critic->calc_update(_network_critic_gradient->get_gradient());
		_network_critic_gradient->calc_gradient();
		

		_network_actor->activate(&_input_actor_s0);		
		Tensor actor_loss = -_network_critic_gradient->get_input_gradient(_sample_size, _network_critic->get_input_dim() - _network_actor->get_output_dim(), _network_actor->get_output_dim());
		_network_actor_gradient->calc_gradient(&actor_loss);
		_update_rule_actor->calc_update(_network_actor_gradient->get_gradient());

		_network_critic->update(_update_rule_critic->get_update());
		_network_actor->update(_update_rule_actor->get_update());

		_network_critic_target->polyak_averaging(0.95, _network_critic);
		_network_actor_target->polyak_averaging(0.95, _network_actor);
	}


	return error;
}

Tensor DDPG::get_action(Tensor* p_state, const float p_step)
{
	ou_process();
	_sigma = _max_sigma - (_max_sigma - _min_sigma) * min(1.0f, p_step / _decay_period);

	Tensor output({ _network_actor->get_output_dim() }, Tensor::ZERO);
	_network_actor->activate(p_state);

	TensorOperator::instance().vv_add(_network_actor->get_output()->arr(), _ou_state->arr(), output.arr(), output.size());
	/*
	Tensor output({ _network_actor->get_output_dim() }, Tensor::ZERO);
	_network_actor->activate(p_state);

	for (int i = 0; i < _network_actor->get_output_dim(); i++)
	{
		const float rand = RandomGenerator::get_instance().normal_random(0, 1);
		output[i] = _network_actor->get_output()->at(i) + rand;
	}
	*/

	return output;
}

void DDPG::reset() const
{
	_ou_state->fill(1);
	TensorOperator::instance().vc_prod(_ou_state->arr(), _mu, _ou_state->arr(), _ou_state->size());
}

void DDPG::ou_process() const
{
	for(int i = 0; i < _ou_state->size(); i++)
	{
		(*_ou_state)[i] += _theta * (_mu - (*_ou_state)[i]) + RandomGenerator::get_instance().normal_random(0, _sigma);
	}
}