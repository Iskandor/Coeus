#include "DeepQLearning.h"

using namespace Coeus;

DeepQLearning::DeepQLearning(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm, const float p_gamma, const int p_size, const int p_sample)
{
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_gamma = p_gamma;

	_replay_buffer = new ReplayBuffer(p_size);
	_sample_size = p_sample;

	_target = new vector<Tensor*>(_sample_size);
	_input = new vector<Tensor*>(_sample_size);

	for(int i = 0; i < _sample_size; i++) {
		(*_target)[i] = new Tensor({ _network->get_output()->size() }, Tensor::ZERO);
		(*_input)[i] = new Tensor({ _network->get_input()[0]->size() }, Tensor::ZERO);
	}
}

DeepQLearning::~DeepQLearning()
{
	delete _replay_buffer;
}

float DeepQLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const float p_reward, const bool p_final) const {
	_replay_buffer->add_item(p_state0, p_action0, p_state1, p_reward, p_final);

	float error = 0;

	if (_replay_buffer->get_size() >= _sample_size) {
		vector<ReplayBuffer::Item*>* sample = _replay_buffer->get_sample(_sample_size);

		for (int i = 0; i < sample->size(); i++) {
			const float maxQs1a = calc_max_qa(&sample->at(i)->s1);

			_network->activate(&sample->at(i)->s0);
			_input->at(i)->override(&sample->at(i)->s0);
			_target->at(i)->override(_network->get_output());
			if (sample->at(i)->final) {
				_target->at(i)->set(sample->at(i)->a, sample->at(i)->r);
			}
			else {
				_target->at(i)->set(sample->at(i)->a, sample->at(i)->r + _gamma * maxQs1a);
			}			
		}

		error = _gradient_algorithm->train(_input, _target, sample->size());
	}

	return error;
}

float DeepQLearning::calc_max_qa(Tensor* p_state) const {
	_network->activate(p_state);
	const int maxQa = _network->get_output()->max_value_index();

	return _network->get_output()->at(maxQa);
}
