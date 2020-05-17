#include "tensor_initializer.h"
#include "random_generator.h"


tensor_initializer::tensor_initializer(const TYPE p_type, const float p_arg1, const float p_arg2) :
	_type(p_type),
	_arg1(p_arg1),
	_arg2(p_arg2)
{
	
}

tensor_initializer::tensor_initializer(tensor_initializer& p_copy) :
	_type(p_copy._type),
	_arg1(p_copy._arg1),
	_arg2(p_copy._arg2)
{
	
}

tensor_initializer* tensor_initializer::uniform(const float p_lower_bound, const float p_upper_bound)
{
	return new tensor_initializer(UNIFORM, p_lower_bound, p_upper_bound);
}

tensor_initializer* tensor_initializer::normal(const float p_mean, const float p_sigma)
{
	return new tensor_initializer(NORMAL, p_mean, p_sigma);
}

tensor_initializer* tensor_initializer::lecun_uniform()
{
	return new tensor_initializer(LECUN_UNIFORM, 0.f, 0.f);
}

tensor_initializer* tensor_initializer::lecun_normal()
{
	return new tensor_initializer(LECUN_NORMAL, 0.f, 0.f);
}

tensor_initializer* tensor_initializer::glorot_uniform()
{
	return new tensor_initializer(GLOROT_UNIFORM, 0.f, 0.f);
}

tensor_initializer* tensor_initializer::glorot_normal()
{
	return new tensor_initializer(GLOROT_NORMAL, 0.f, 0.f);
}

tensor_initializer* tensor_initializer::xavier_uniform()
{
	return new tensor_initializer(XAVIER_UNIFORM, 0.f, 0.f);
}

tensor_initializer* tensor_initializer::xavier_normal()
{
	return new tensor_initializer(XAVIER_NORMAL, 0.f, 0.f);
}

tensor_initializer* tensor_initializer::debug(const float p_value)
{
	return new tensor_initializer(DEBUG, p_value, 0.f);
}

tensor_initializer::~tensor_initializer()
= default;

void tensor_initializer::init(tensor& p_tensor) const
{
	switch(_type)
	{
	case DEBUG:
		p_tensor.fill(_arg1);
		break;
	case UNIFORM:
		init_uniform(p_tensor, _arg1, _arg2);
		break;
	case NORMAL:
		init_normal(p_tensor, _arg1, _arg2);
		break;
	case LECUN_UNIFORM:
		init_uniform(p_tensor, -sqrt(3.f / p_tensor.shape(0)), sqrt(3.f / p_tensor.shape(0)));
		break;
	case LECUN_NORMAL:
		init_normal(p_tensor, 0.f, sqrt(1.f / p_tensor.shape(0)));
		break;
	case GLOROT_UNIFORM:
		init_uniform(p_tensor, -2.f / (p_tensor.shape(0) + p_tensor.shape(1)), 2.f / (p_tensor.shape(0) + p_tensor.shape(1)));
		break;
	case GLOROT_NORMAL:
		init_normal(p_tensor, 0.f, 2.f / (p_tensor.shape(0) + p_tensor.shape(1)));
		break;
	case XAVIER_UNIFORM:
		init_uniform(p_tensor, -sqrt(6.f / (p_tensor.shape(0) + p_tensor.shape(1))), sqrt(6.f / (p_tensor.shape(0) + p_tensor.shape(1))));
		break;
	case XAVIER_NORMAL:
		init_normal(p_tensor, 0.f, sqrt(3.f / (p_tensor.shape(0) + p_tensor.shape(1))));
		break;
	default: ;
	}
}

void tensor_initializer::init_uniform(tensor& p_tensor, const float p_lower_bound, const float p_upper_bound)
{
	float *x = p_tensor._data;
	for (int i = 0; i < p_tensor._size; i++)
	{
		*x++ = random_generator::instance().random(p_lower_bound, p_upper_bound);
	}
}

void tensor_initializer::init_normal(tensor& p_tensor, const float p_mean, const float p_sigma)
{
	float *x = p_tensor._data;
	for (int i = 0; i < p_tensor._size; i++)
	{
		*x++ = random_generator::instance().normal_random(p_mean, p_sigma);
	}
}
