#include "activation_functions.h"
#include "CLAB.h"

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

	_input = forward(_input);

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

			xx = _mm256_mul_ps(_mm256_sub_ps(plus_onex, xx), xx);
			dx = _mm256_mul_ps(dx, xx);

			_mm256_storeu_ps(pd, dx);
			px += segment;
			pd += segment;
		}
	}

	for (int i = size * segment; i < p_delta.size(); i++) {
		*pd = *pd * *px * (1 - *px);
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

	_input = forward(_input);

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

			xx = _mm256_sub_ps(plus_onex, _mm256_mul_ps(xx, xx));
			dx = _mm256_mul_ps(dx, xx);

			_mm256_storeu_ps(pd, dx);
			px += segment;
			pd += segment;
		}
	}

	for (int i = size * segment; i < p_delta.size(); i++) {
		*pd = *pd * 1 - *px * *px;
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

	_input = forward(_input);

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
		*pd = *pd * (std::tanh(exp(*px)) - *px * exp(*px) * (std::tanh(exp(*px)) * std::tanh(exp(*px)) - 1));
		pd++;
		px++;
	}

	return p_delta;
}
