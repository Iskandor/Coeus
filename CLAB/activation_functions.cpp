#include "activation_functions.h"
#include "CLAB.h"
#include <algorithm>

activation_function* activation_function::linear()
{
	return new linear_function();
}

activation_function* activation_function::sigmoid()
{
	return new sigmoid_function();
}

activation_function* activation_function::tanh()
{
	return new tanh_function();
}

activation_function* activation_function::tanhexp()
{
	return new tanhexp_function();
}

activation_function* activation_function::relu()
{
	return new relu_function();
}

activation_function* activation_function::softmax()
{
	return new softmax_function();
}

activation_function* activation_function::create(const TYPE p_type)
{
	activation_function* result = nullptr;
	switch(p_type)
	{
	case LINEAR:
		result = new linear_function();
		break;
	case SIGMOID:
		result = new sigmoid_function();
		break;
	case TANH:
		result = new tanh_function();
		break;
	case TANHEXP:
		result = new tanhexp_function();
		break;
	case RELU:
		result = new relu_function();
		break;
	default: ;
	}

	return result;
}

tensor& linear_function::forward(tensor& p_input)
{
	return p_input;
}

tensor& linear_function::backward(tensor& p_delta)
{
	return p_delta;
}

tensor& sigmoid_function::forward(tensor& p_input)
{	
	_input = p_input;
	const int size = p_input.size() / segment;
	float* px = p_input.data();

	if (size > 0)
	{
		const float plus_one = 1.0f;
		const float minus_one = -1.0f;
		const __m256 plus_onex = _mm256_broadcast_ss(&plus_one);
		const __m256 minus_onex = _mm256_broadcast_ss(&minus_one);

		for (int i = 0; i < size; i++)
		{
			__m256 xx = _mm256_load_ps(px);
			xx = _mm256_mul_ps(xx, minus_onex);
			for(int j = 0; j < segment; j++)
			{
				xx.m256_f32[j] = exp(xx.m256_f32[j]);
			}
			xx = _mm256_add_ps(plus_onex, xx);
			xx = _mm256_div_ps(plus_onex, xx);
			_mm256_storeu_ps(px, xx);
			px += segment;
		}
	}

	for(int i = size * segment; i < p_input.size(); i++)
	{
		*px = 1 / (1 + exp(-*px));
		px++;
	}

	return p_input;
}

tensor& sigmoid_function::backward(tensor& p_delta)
{
	const int size = p_delta.size() / segment;

	tensor output = _input;

	float* px = forward(output).data();
	float* pd = p_delta.data();

	if (size > 0)
	{
		const float plus_one = 1.0f;
		const __m256 plus_onex = _mm256_broadcast_ss(&plus_one);

		for (int i = 0; i < size; i++)
		{
			__m256 dx = _mm256_load_ps(pd);
			__m256 xx = _mm256_load_ps(px);

			xx = _mm256_mul_ps(_mm256_sub_ps(plus_onex, xx), xx);
			dx = _mm256_mul_ps(dx, xx);

			_mm256_storeu_ps(pd, dx);
			px += segment;
			pd += segment;
		}
	}

	for (int i = size * segment; i < p_delta.size(); i++) {
		*pd *= *px * (1 - *px);
		pd++;
		px++;
	}

	return p_delta;
}

tensor& tanh_function::forward(tensor& p_input)
{
	_input = p_input;
	float* px = p_input.data();
	for (int i = 0; i < p_input.size(); i++)
	{
		*px = std::tanh(*px);
		px++;
	}

	return p_input;
}

tensor& tanh_function::backward(tensor& p_delta)
{
	const int size = p_delta.size() / segment;

	tensor output = _input;

	float* px = forward(output).data();
	float* pd = p_delta.data();

	if (size > 0)
	{
		const float plus_one = 1.0f;
		const __m256 plus_onex = _mm256_broadcast_ss(&plus_one);

		for (int i = 0; i < size; i++)
		{
			__m256 dx = _mm256_load_ps(pd);
			const __m256 xx = _mm256_load_ps(px);

			dx = _mm256_mul_ps(dx, _mm256_sub_ps(plus_onex, _mm256_mul_ps(xx, xx)));

			_mm256_storeu_ps(pd, dx);
			px += segment;
			pd += segment;
		}
	}

	for (int i = size * segment; i < p_delta.size(); i++) {
		*pd *= 1 - *px * *px;
		pd++;
		px++;
	}

	return p_delta;
}

tensor& tanhexp_function::forward(tensor& p_input)
{
	_input = p_input;
	const int size = p_input.size() / segment;
	float* px = p_input.data();

	if (size > 0)
	{
		for (int i = 0; i < size; i++)
		{
			__m256 xx = _mm256_load_ps(px);
			__m256 expx = _mm256_load_ps(px);
			for (int j = 0; j < segment; j++)
			{
				expx.m256_f32[j] = std::tanh(exp(expx.m256_f32[j]));
			}
			xx = _mm256_mul_ps(xx, expx);
			_mm256_storeu_ps(px, xx);
			px += segment;
		}
	}

	for (int i = size * segment; i < p_input.size(); i++)
	{
		*px = *px * std::tanh(exp(*px));
		px++;
	}

	return p_input;
}

tensor& tanhexp_function::backward(tensor& p_delta)
{
	const int size = p_delta.size() / segment;

	float* px = _input.data();
	float* pd = p_delta.data();

	if (size > 0)
	{
		const float plus_one = 1.0f;
		const __m256 plus_onex = _mm256_broadcast_ss(&plus_one);

		for (int i = 0; i < size; i++)
		{
			__m256 dx = _mm256_load_ps(pd);
			__m256 xx = _mm256_load_ps(px);
			__m256 ex;
			__m256 tanhex;

			for (int j = 0; j < segment; j++)
			{
				ex.m256_f32[j] = exp(xx.m256_f32[j]);
				tanhex.m256_f32[j] = std::tanh(ex.m256_f32[j]);
				if (tanhex.m256_f32[j] == 1.f) ex.m256_f32[j] = 0.f;
			}
			__m256 xx2;
			xx2 = _mm256_mul_ps(tanhex, tanhex);
			xx2 = _mm256_sub_ps(xx2, plus_onex);
			xx2 = _mm256_mul_ps(ex, xx2);
			xx2 = _mm256_mul_ps(xx, xx2);
			xx = _mm256_sub_ps(tanhex, xx2);
			dx = _mm256_mul_ps(dx, xx);

			_mm256_storeu_ps(pd, dx);
			px += segment;
			pd += segment;
		}
	}

	for (int i = size * segment; i < p_delta.size(); i++) {
		float ex = exp(*px);
		float tanhex = std::tanh(ex);
		if (tanhex < 1.f)
		{
			*pd *= tanhex - *px * ex * (tanhex * tanhex - 1.f);
		}
		pd++;
		px++;
	}

	return p_delta;
}

tensor& relu_function::forward(tensor& p_input)
{
	_input = p_input;

	float* px = p_input.data();
	for (int i = 0; i < p_input.size(); i++)
	{
		*px = std::max(*px, 0.f);
		px++;
	}

	return p_input;
}

tensor& relu_function::backward(tensor& p_delta)
{
	float* px = _input.data();
	float* pd = p_delta.data();

	for (int i = 0; i < p_delta.size(); i++) {
		if (*px < 0.f) *pd *= 0.f;
		pd++;
		px++;
	}

	return p_delta;
}

tensor& softmax_function::forward(tensor& p_input)
{
	_input = p_input;

	float* px = p_input.data();

	for (int i = 0; i < p_input.shape(0); i++)
	{
		float esum = 0;
		float max = 0;

		for (int j = 0; j < p_input.shape(1); j++) {
			if (max < p_input[i * p_input.shape(1) + j])
			{
				max = p_input[i * p_input.shape(1) + j];
			}
		}

		for (int j = 0; j < p_input.shape(1); j++) {
			p_input[i * p_input.shape(1) + j] = exp(p_input[i * p_input.shape(1) + j] - max);
			esum += p_input[i * p_input.shape(1) + j];
		}

		for (int j = 0; j < p_input.shape(1); j++) {
			p_input[i * p_input.shape(1) + j] /= esum;
		}
	}

	return p_input;
}

tensor& softmax_function::backward(tensor& p_delta)
{
	/*
	Tensor tg({ p_input->shape(1) }, Tensor::ZERO);
	Tensor ti({ p_input->shape(1) }, Tensor::ZERO);

	_derivative.resize({ p_delta.shape(1) , p_delta.shape(1) });

	for (int i = 0; i < p_delta.shape(0); i++)
	{
		for (int r = 0; r < p_delta.shape(1); r++) {
			for (int c = 0; c < p_delta.shape(1); c++) {
				_derivative[r * p_delta.shape(1) + c] = _input[i * p_delta.shape(1) + r] * (kronecker_delta(r, c) - _input[i * p_delta.shape(1) + c]);
			}
		}

		p_input->get_row(ti, b);

		TensorOperator::instance().vM_prod(ti.arr(), deriv, tg.arr(), _output->shape(1), _output->shape(1));

		_gradient->push_back(&tg);
	}
	*/

	return p_delta;
}

float softmax_function::kronecker_delta(int p_i, int p_j) const
{
	return p_i == p_j ? 1.f : 0.f;
}
