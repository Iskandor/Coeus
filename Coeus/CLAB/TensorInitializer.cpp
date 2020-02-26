#include "TensorInitializer.h"
#include "RandomGenerator.h"

using namespace Coeus;

TensorInitializer::TensorInitializer()
{
	_init = DEBUG;
	_arg1 = 0;
	_arg2 = 0;
}

TensorInitializer::TensorInitializer(TensorInitializer& p_copy)
{
	_init = p_copy._init;
	_arg1 = p_copy._arg1;
	_arg2 = p_copy._arg2;
}


TensorInitializer::TensorInitializer(const INIT p_init, const float p_arg1, const float p_arg2)
{
	_init = p_init;
	_arg1 = p_arg1;
	_arg2 = p_arg2;
}


TensorInitializer::~TensorInitializer()
= default;

void TensorInitializer::init(Tensor* p_tensor, const INIT p_init, const float p_arg1, const float p_arg2)
{
	int in_dim = 0; 
	int out_dim = 0;

	if (p_tensor->rank() == 2)
	{
		out_dim = p_tensor->shape(0);
		in_dim = p_tensor->shape(1);
	}

	switch (p_init) {
		case DEBUG:
			p_tensor->fill(1);
			break;
		case UNIFORM:
			uniform(p_tensor, p_arg1, p_arg2);
			break;
		case LECUN_UNIFORM:
			uniform(p_tensor , -sqrt(3.f / in_dim), sqrt(3.f / in_dim));
			break;
		case GLOROT_UNIFORM:
			uniform(p_tensor , -2.f / (in_dim + out_dim), 2.f / (in_dim + out_dim));
			break;
		case IDENTITY:
			p_tensor->fill(Tensor::ONES);
			break;
		case NORMAL:
			normal(p_tensor, p_arg1, p_arg2);
			break;
		case EXPONENTIAL:
			exponential(p_tensor, p_arg1);
			break;
		case HE_UNIFORM:
			uniform(p_tensor , -sqrt(6.f / in_dim), sqrt(6.f / in_dim));
			break;
		case LECUN_NORMAL:
			normal(p_tensor, 0., sqrt(1.f / in_dim));
			break;
		case GLOROT_NORMAL:
			normal(p_tensor, 0., 2.f / (in_dim + out_dim));
			break;
		case HE_NORMAL:
			normal(p_tensor, 0., sqrt(2.f / in_dim));
			break;
	}
}

void TensorInitializer::init(Tensor* p_tensor) const
{
	init(p_tensor, _init, _arg1, _arg2);
}


void TensorInitializer::uniform(Tensor* p_tensor, const float p_min, const float p_max) {
	float* x = &p_tensor->arr()[0];

	for (int i = 0; i < p_tensor->size(); i++)
	{
		*x++ = RandomGenerator::get_instance().random(p_min, p_max);
	}
}

void TensorInitializer::normal(Tensor* p_tensor, const float p_mean, const float p_dev)
{
	float* x = &p_tensor->arr()[0];

	for (int i = 0; i < p_tensor->size(); i++)
	{
		*x++ = RandomGenerator::get_instance().normal_random(p_mean, p_dev);
	}
}

void TensorInitializer::exponential(Tensor* p_tensor, const float p_lambda)
{
	float* x = &p_tensor->arr()[0];

	for (int i = 0; i < p_tensor->size(); i++)
	{
		*x++ = RandomGenerator::get_instance().exp_random(p_lambda);
	}
}