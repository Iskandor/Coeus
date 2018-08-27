#include "ICM.h"
#include "Encoder.h"

using namespace Coeus;

ICM::ICM(NeuralNetwork* p_forward_model, BaseGradientAlgorithm* p_forward_alogrithm, NeuralNetwork* p_inverse_model, BaseGradientAlgorithm* p_inverse_alogrithm) {
	_forward_model = p_forward_model;
	_forward_alogrithm = p_forward_alogrithm;
	_inverse_model = p_inverse_model;
	_inverse_alogrithm = p_inverse_alogrithm;
	_forward_reward = _inverse_reward = 0;

	_forward_model_input = nullptr;
	_inverse_model_input = nullptr;
}

ICM::~ICM()
{
	if (_forward_model_input != nullptr) delete _forward_model_input;
	if (_inverse_model_input != nullptr) delete _inverse_model_input;
}

double ICM::train(Tensor* p_state0, const int p_action0, Tensor* p_state1) {
	
	double error = 0;

	if (_forward_model_input == nullptr) {
		_forward_model_input = new Tensor({ _inverse_model->get_output()->size() + p_state0->size() }, Tensor::ZERO);
	}
	if (_inverse_model_input == nullptr) {
		_inverse_model_input = new Tensor({ p_state0->size() + p_state1->size() }, Tensor::ZERO);
	}

	Tensor action({ _inverse_model->get_output()->size() }, Tensor::ZERO);

	Encoder::one_hot(action, p_action0);

	Tensor::Concat(_forward_model_input, &action, p_state0);
	Tensor::Concat(_inverse_model_input, p_state0, p_state1);

	_forward_model->activate(_forward_model_input);
	_inverse_model->activate(_inverse_model_input);


	_forward_reward = _L.cost(_forward_model->get_output(), p_state1);
	_inverse_reward = _L.cost(_inverse_model->get_output(), &action);


	error = _forward_alogrithm->train(_forward_model_input, p_state1) + _inverse_alogrithm->train(_inverse_model_input, &action);

	return error;
}

double ICM::get_intrinsic_reward(const double p_beta, const double p_eta) const {
	return (1 - p_beta) * _inverse_reward + p_beta * p_eta *  _forward_reward;
}

