#include "SoftmaxActivation.h"
#include <cmath>
#include "TensorOperator.h"

using namespace Coeus;

SoftmaxActivation::SoftmaxActivation(): IActivationFunction(SOFTMAX) {
}


SoftmaxActivation::~SoftmaxActivation()
{
}

Tensor* SoftmaxActivation::backward(Tensor* p_input, Tensor* p_x)
{
	IActivationFunction::backward(p_input, p_x);
	float* deriv = Tensor::alloc_arr(_output->size() * _output->size());

	if (p_x == nullptr)
	{
		for (int i = 0; i < _output->size(); i++) {
			for (int j = 0; j < _output->size(); j++) {
				deriv[i * _output->size() + j] = (*_output)[i] * (Tensor::kronecker_delta(i, j) - (*_output)[j]);
			}
		}
	}
	else
	{
		for (int i = 0; i < p_x->size(); i++) {
			for (int j = 0; j < p_x->size(); j++) {
				deriv[i * p_x->size() + j] = (*p_x)[i] * (Tensor::kronecker_delta(i, j) - (*p_x)[j]);
			}
		}
	}

	TensorOperator::instance().vM_prod(p_input->arr(), deriv, _gradient->arr(), _output->size(), _output->size());

	delete[] deriv;

	return _gradient;
}

Tensor* SoftmaxActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);
	float esum = 0;
	const float max = (*_input)[_input->max_value_index()];

	for (int i = 0; i < _output->size(); i++) {
		(*_output)[i] = exp((*_input)[i] - max);
		esum += (*_output)[i];
	}

	for (int i = 0; i < _output->size(); i++) {
		(*_output)[i] /= esum;
	}

	return _output;
}

Tensor SoftmaxActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size() * p_input.size());

	Tensor* activation = forward(&p_input);

	for (int i = 0; i < p_input.size(); i++) {
		for (int j = 0; j < p_input.size(); j++) {
			arr[i * p_input.size() + j] = (*activation)[i] * (Tensor::kronecker_delta(i, j) - (*activation)[j]);
		}
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}