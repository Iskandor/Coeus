#include "DDPG.h"

using namespace Coeus;


DDPG::DDPG(	NeuralNetwork* p_network_critic, BaseGradientAlgorithm* p_gradient_algorithm_critic, const double p_gamma, 
			NeuralNetwork* p_network_actor, BaseGradientAlgorithm* p_gradient_algorithm_actor,
           	const int p_buffer_size, const int p_sample_size):
	_network_actor(p_network_actor), _network_critic(p_network_critic),	_gradient_algorithm_actor(p_gradient_algorithm_actor), _gradient_algorithm_critic(p_gradient_algorithm_critic),
	_gamma(p_gamma), _sample_size(p_sample_size) {

	_network_actor_target = new NeuralNetwork(*p_network_actor);
	_network_critic_target = new NeuralNetwork(*p_network_critic);

	_buffer = new ReplayBuffer(p_buffer_size);
}

DDPG::~DDPG()
{
}

double DDPG::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const double p_reward) {
	_buffer->add_item(p_state0, p_action0, p_state1, p_reward, false);

	double error = 0;

	if (_buffer->get_size() >= _sample_size) {
		vector<ReplayBuffer::Item*>* sample = _buffer->get_sample(_sample_size);

		_input.clear();
		_target.clear();

		for (int i = 0; i < sample->size(); i++) {
			const double maxQs1a = calc_max_qa(&sample->at(i)->s1);

			_network_critic->activate(&sample->at(i)->s0);
			_input.push_back(&sample->at(i)->s0);
			_target.push_back(_network_critic->get_output());
			(*_target.end())->set(sample->at(i)->a, sample->at(i)->r + _gamma * maxQs1a);
		}

		error = _gradient_algorithm_critic->train(&_input, &_target);
	}

	return error;
}

double DDPG::calc_max_qa(Tensor* p_state) const {
	_network_critic_target->activate(p_state);
	_network_actor_target->activate(p_state);

	const int action = _network_actor_target->get_output()->max_value_index();

	return _network_critic_target->get_output()->at(action);
}
