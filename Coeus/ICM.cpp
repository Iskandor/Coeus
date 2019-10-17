#include "ICM.h"
#include "Encoder.h"

using namespace Coeus;

ICM::ICM(NeuralNetwork* p_forward_model, GradientAlgorithm* p_forward_algorithm, const int p_size) {
	_forward_model = p_forward_model;
	_forward_algorithm = p_forward_algorithm;
	_forward_reward = 0;

	_buffer = nullptr;

	if (p_size > 0)
	{
		_buffer = new ReplayBuffer<TransitionItem>(p_size);
	}

	_input = nullptr;
	_target = nullptr;
}

ICM::~ICM()
{
	delete _buffer;
	delete _input;
	delete _target;
}

void ICM::activate(Tensor* p_state0, Tensor* p_action)
{
	if (_input == nullptr || _input->size() != _forward_model->get_input_dim()) {
		delete _input;
		_input = new Tensor({ _forward_model->get_input_dim() }, Tensor::ZERO);
	}

	_input->reset_index();
	_input->push_back(p_action);
	_input->push_back(p_state0);

	_forward_model->activate(_input);
}

float ICM::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) {

	activate(p_state0, p_action);
	const float error = _forward_algorithm->train(_input, p_state1);

	return error;
}

void ICM::add(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) const
{
	if (_buffer != nullptr)
	{
		_buffer->add_item(new TransitionItem(p_state0, p_action, p_state1));
	}

}

float ICM::train(const int p_sample)
{
	float error = 0;
	if (_buffer->get_size() >= p_sample) {

		if (_input == nullptr || _input->shape(0) != p_sample)
		{
			delete _input;
			_input = new Tensor({ p_sample, _forward_model->get_input_dim() }, Tensor::ZERO);
		}
		if (_target == nullptr || _target->shape(0) != p_sample)
		{
			delete _target;
			_target = new Tensor({ p_sample, _forward_model->get_output_dim() }, Tensor::ZERO);
		}

		vector<TransitionItem*>* sample = _buffer->get_sample(p_sample);

		_input->reset_index();
		_target->reset_index();

		for (auto& i : *sample)
		{
			_input->push_back(&i->a);
			_input->push_back(&i->s0);
			_target->push_back(&i->s1);
		}

		error = _forward_algorithm->train(_input, _target);
	}

	return error;
}

float ICM::get_intrinsic_reward(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, const float p_eta) {
	activate(p_state0, p_action);
	_forward_reward = _L.cost(_forward_model->get_output(), p_state1);
	return p_eta *  _forward_reward;
}

