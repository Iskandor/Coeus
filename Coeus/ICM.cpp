#include "ICM.h"
#include "Encoder.h"

using namespace Coeus;

ICM::ICM(NeuralNetwork* p_forward_model, GradientAlgorithm* p_forward_alogrithm) {
	_forward_model = p_forward_model;
	_forward_alogrithm = p_forward_alogrithm;
	_forward_reward = 0;

	_forward_model_input = nullptr;
}

ICM::~ICM()
{
	if (_forward_model_input != nullptr) delete _forward_model_input;
}

float ICM::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1) {
	
	if (_forward_model_input == nullptr) {
		_forward_model_input = new Tensor({ p_action->size() + p_state0->size() }, Tensor::ZERO);
	}

	Tensor::concat(_forward_model_input, p_action, p_state0);

	_forward_model->activate(_forward_model_input);
	_forward_reward = _L.cost(_forward_model->get_output(), p_state1);


	const float error = _forward_alogrithm->train(_forward_model_input, p_state1);

	return error;
}

float ICM::get_intrinsic_reward(const float p_eta) const {
	return p_eta *  _forward_reward;
}

