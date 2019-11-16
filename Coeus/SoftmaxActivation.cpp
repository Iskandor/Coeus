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

	if (p_input->rank() == 1)
	{

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
	}
	if (p_input->rank() == 2)
	{
		Tensor tg({ p_input->shape(1) }, Tensor::ZERO);
		Tensor ti({ p_input->shape(1) }, Tensor::ZERO);

		_gradient->reset_index();
		float* deriv = Tensor::alloc_arr(p_input->shape(1) * p_input->shape(1));

		for (int b = 0; b < p_input->shape(0); b++)
		{
			if (p_x == nullptr)
			{
				for (int i = 0; i < p_input->shape(1); i++) {
					for (int j = 0; j < p_input->shape(1); j++) {
						deriv[i * p_input->shape(1) + j] = _output->at(b, i) * (Tensor::kronecker_delta(i, j) - _output->at(b, j));
					}
				}
			}
			else
			{
				for (int i = 0; i < p_x->shape(1); i++) {
					for (int j = 0; j < p_x->shape(1); j++) {
						deriv[i * p_x->shape(1) + j] = p_x->at(b, i) * (Tensor::kronecker_delta(i, j) - p_x->at(b, j));
					}
				}
			}

			p_input->get_row(ti, b);

			TensorOperator::instance().vM_prod(ti.arr(), deriv, tg.arr(), _output->shape(1), _output->shape(1));

			_gradient->push_back(&tg);
		}
		delete[] deriv;
	}

	return _gradient;
}

Tensor* SoftmaxActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);

	if (p_input->rank() == 1)
	{
		float esum = 0;
		const float max = (*_input)[_input->max_value_index()];

		for (int i = 0; i < _output->size(); i++) {
			(*_output)[i] = exp((*_input)[i] - max);
			esum += (*_output)[i];
		}

		for (int i = 0; i < _output->size(); i++) {
			(*_output)[i] /= esum;
		}
	}
	if (p_input->rank() == 2)
	{
		for(int b = 0; b < p_input->shape(0); b++)
		{
			float esum = 0;
			float max = 0;

			for (int i = 0; i < p_input->shape(1); i++) {
				if (max < _input->at(b, i))
				{
					max = _input->at(b, i);
				}
			}

			for (int i = 0; i < p_input->shape(1); i++) {
				_output->set(b, i, exp(_input->at(b, i) - max));
				esum += _output->at(b, i);
			}

			for (int i = 0; i < p_input->shape(1); i++) {
				_output->set(b, i, _output->at(b, i) / esum);
			}
		}
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