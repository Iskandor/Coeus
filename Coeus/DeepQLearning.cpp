#include "DeepQLearning.h"

using namespace Coeus;

DeepQLearning::DeepQLearning(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm, const float p_gamma, const int p_size, const int p_sample)
{
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_gamma = p_gamma;

	_replay_buffer = new ReplayBuffer<DQItem>(p_size);
	_sample_size = p_sample;

	_target = new Tensor({ p_sample, _network->get_output_dim() }, Tensor::ZERO);
	_input = new Tensor({ p_sample, _network->get_input_dim() }, Tensor::ZERO);
}

DeepQLearning::~DeepQLearning()
{
	delete _replay_buffer;
}

float DeepQLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const float p_reward, const bool p_final) const {
	_replay_buffer->add_item(new DQItem(p_state0, p_action0, p_state1, p_reward, p_final));

	float error = 0;

	if (_replay_buffer->get_size() >= _sample_size) {
		vector<DQItem*>* sample = _replay_buffer->get_sample(_sample_size);

		_input->reset_index();
		_target->reset_index();

		for (int i = 0; i < sample->size(); i++) {
			const float maxQs1a = calc_max_qa(&sample->at(i)->s1);

			_network->activate(&sample->at(i)->s0);
			_input->push_back(&sample->at(i)->s0);
			
			Tensor *target = _network->get_output();

			if (sample->at(i)->final) {
				target->set(sample->at(i)->a, sample->at(i)->r);
			}
			else {
				target->set(sample->at(i)->a, sample->at(i)->r + _gamma * maxQs1a);
			}

			_target->push_back(target);
		}

		error = _gradient_algorithm->train(_input, _target);
	}

	return error;
}

float DeepQLearning::calc_max_qa(Tensor* p_state) const {
	_network->activate(p_state);
	const int maxQa = _network->get_output()->max_value_index();

	return _network->get_output()->at(maxQa);
}
