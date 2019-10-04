#include "DDPG.h"

using namespace Coeus;


DDPG::DDPG(	NeuralNetwork* p_network_critic, GradientAlgorithm* p_gradient_algorithm_critic, const float p_gamma, 
			NeuralNetwork* p_network_actor, GradientAlgorithm* p_gradient_algorithm_actor,
           	const int p_buffer_size, const int p_sample_size):
	_network_actor(p_network_actor), _network_critic(p_network_critic),	_gradient_algorithm_actor(p_gradient_algorithm_actor), _gradient_algorithm_critic(p_gradient_algorithm_critic),
	_gamma(p_gamma), _sample_size(p_sample_size) {

	_network_actor_target = new NeuralNetwork(*p_network_actor);
	_network_critic_target = new NeuralNetwork(*p_network_critic);

	_buffer = new ReplayBuffer<DQItem>(p_buffer_size);

	_target = new Tensor({ p_sample_size, p_network_critic->get_output_dim() }, Tensor::ZERO);
	_input = new Tensor({ p_sample_size, p_network_critic->get_input_dim() }, Tensor::ZERO);

}

DDPG::~DDPG()
{
	delete _buffer;
}

float DDPG::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const float p_reward) {
	_buffer->add_item(new DQItem(p_state0, p_action0, p_state1, p_reward, false));

	float error = 0;

	if (_buffer->get_size() >= _sample_size) {
		vector<DQItem*>* sample = _buffer->get_sample(_sample_size);

		_input->reset_index();
		_target->reset_index();

		for (int i = 0; i < sample->size(); i++) {
			const float maxQs1a = calc_max_qa(&sample->at(i)->s1);

			_network_critic->activate(&sample->at(i)->s0);
			_input->push_back(&sample->at(i)->s0);

			Tensor *target = _network_critic->get_output();

			if (sample->at(i)->final) {
				target->set(sample->at(i)->a, sample->at(i)->r);
			}
			else {
				target->set(sample->at(i)->a, sample->at(i)->r + _gamma * maxQs1a);
			}

			_target->push_back(target);
		}

		error = _gradient_algorithm_critic->train(_input, _target);
	}

	return error;
}

float DDPG::calc_max_qa(Tensor* p_state) const {
	_network_critic_target->activate(p_state);
	_network_actor_target->activate(p_state);

	const int action = _network_actor_target->get_output()->max_value_index();

	return _network_critic_target->get_output()->at(action);
}
